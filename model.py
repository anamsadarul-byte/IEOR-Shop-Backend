import pyomo.environ as am
import pandas as pd
from datetime import datetime


def run_model(forecast_file, inventory_file, shelf_file, delivery_file):

    # ==========================================
    # 1. LOAD FILES
    # ==========================================
    df_forecast = pd.read_csv(forecast_file)
    df_inv = pd.read_csv(inventory_file)
    df_shelf = pd.read_csv(shelf_file)
    df_delivery = pd.read_csv(delivery_file)

    # ==========================================
    # 2. PREPROCESS FORECAST DATA
    # ==========================================
    df_forecast['clean_date'] = (
        df_forecast['forecast_date']
        .astype(str).str.replace("'", "").str.strip()
    )

    ITEMS = df_forecast['item_code'].unique().tolist()
    unique_dates = sorted(df_forecast['clean_date'].unique().tolist())

    T_MAP = {i+1: date for i, date in enumerate(unique_dates)}
    REVERSE_T_MAP = {date: i+1 for i, date in enumerate(unique_dates)}

    D_data = {(i, t): 0.0 for i in ITEMS for t in T_MAP.keys()}

    for _, row in df_forecast.iterrows():
        i = row['item_code']
        date_str = row['clean_date']
        qty = row['forecast_quantity']
        if date_str in REVERSE_T_MAP:
            t = REVERSE_T_MAP[date_str]
            D_data[(i, t)] = float(qty)

    # ==========================================
    # 3. SUPPORTING FILES
    # ==========================================
    L_data = dict(zip(df_shelf['item_code'], df_shelf['item_shelf_life']))
    for i in ITEMS:
        if i not in L_data:
            L_data[i] = 5

    df_inv['date'] = df_inv['date'].astype(str).str.replace("'", "").str.strip()

    InitInv_data = {}
    for i in ITEMS:
        record = df_inv[df_inv['item_code'] == i]
        InitInv_data[i] = float(record['onhand_quantity'].values[0]) if not record.empty else 0.0

    delta_data = {}
    for t, date_str in T_MAP.items():
        dt_obj = datetime.strptime(date_str, '%Y-%m-%d')
        day_name = dt_obj.strftime('%a')
        for i in ITEMS:
            row = df_delivery[df_delivery['item_code'] == i]
            if row.empty or day_name not in df_delivery.columns:
                delta_data[(i,t)] = 0
            else:
                val = row.iloc[0][day_name]
                delta_data[(i,t)] = 1 if str(val).upper() == 'Y' else 0

    # ==========================================
    # 4. BUILD MODEL
    # ==========================================
    model = am.ConcreteModel()

    model.S = am.Set(initialize=ITEMS)
    model.T = am.Set(initialize=T_MAP.keys())
    model.A = am.RangeSet(1, max(L_data.values()))

    model.D = am.Param(model.S, model.T, initialize=D_data, default=0)
    model.delta = am.Param(model.S, model.T, initialize=delta_data, default=0)
    model.InitInv = am.Param(model.S, initialize=InitInv_data, default=0)
    model.L_life = am.Param(model.S, initialize=L_data)

    model.Qmin = am.Param(initialize=80)
    model.Qmax = am.Param(initialize=150)
    model.M = am.Param(initialize=100)

    model.Q = am.Var(model.S, model.T, domain=am.NonNegativeIntegers)
    model.X = am.Var(model.S, model.A, model.T, domain=am.NonNegativeReals)
    model.I = am.Var(model.S, model.A, model.T, domain=am.NonNegativeReals)
    model.S_var = am.Var(model.S, model.T, domain=am.NonNegativeReals)
    model.W = am.Var(model.S, model.T, domain=am.NonNegativeReals)
    model.z = am.Var(model.S, model.A, model.T, domain=am.Binary)
    model.y_day = am.Var(model.T, domain=am.Binary)
    model.R = am.Var(domain=am.NonNegativeReals)

    # ==========================================
    # OBJECTIVE
    # ==========================================
    def obj_rule(m):
        daily = sum(m.S_var[i,t] + m.W[i,t] for i in m.S for t in m.T)
        return daily + m.R
    model.obj = am.Objective(rule=obj_rule, sense=am.minimize)

    # ==========================================
    # CONSTRAINTS
    # ==========================================
    def demand_rule(m,i,t):
        valid_usage = sum(m.X[i,a,t] for a in m.A if a <= m.L_life[i])
        return valid_usage + m.S_var[i,t] == m.D[i,t]
    model.c1 = am.Constraint(model.S,model.T,rule=demand_rule)

    def usage_new(m,i,t):
        return m.X[i,1,t] <= m.Q[i,t]
    model.c2 = am.Constraint(model.S,model.T,rule=usage_new)

    def inv_new(m,i,t):
        return m.I[i,1,t] == m.Q[i,t] - m.X[i,1,t]
    model.c3 = am.Constraint(model.S,model.T,rule=inv_new)

    def inv_old(m,i,a,t):
        if a==1 or a>m.L_life[i]:
            return am.Constraint.Skip
        prev = m.InitInv[i] if (t==1 and a==2) else (0 if t==1 else m.I[i,a-1,t-1])
        return m.I[i,a,t] == prev - m.X[i,a,t]
    model.c4 = am.Constraint(model.S,model.A,model.T,rule=inv_old)

    def usage_limit(m,i,a,t):
        if a==1 or a>m.L_life[i]:
            return am.Constraint.Skip
        prev = m.InitInv[i] if (t==1 and a==2) else (0 if t==1 else m.I[i,a-1,t-1])
        return m.X[i,a,t] <= prev
    model.c5 = am.Constraint(model.S,model.A,model.T,rule=usage_limit)

    def wastage(m,i,t):
        L=m.L_life[i]
        return m.W[i,t] == m.I[i,L,t]
    model.c6 = am.Constraint(model.S,model.T,rule=wastage)

    def delivery(m,i,t):
        return m.Q[i,t] <= m.Qmax*m.delta[i,t]
    model.c7 = am.Constraint(model.S,model.T,rule=delivery)

    def qmax(m,t):
        return sum(m.Q[i,t] for i in m.S) <= m.Qmax*m.y_day[t]
    model.c8 = am.Constraint(model.T,rule=qmax)

    def qmin(m,t):
        return sum(m.Q[i,t] for i in m.S) >= m.Qmin*m.y_day[t]
    model.c9 = am.Constraint(model.T,rule=qmin)

    def terminal(m):
        last=max(m.T)
        return m.R == sum(m.I[i,a,last] for i in m.S for a in m.A)
    model.c10 = am.Constraint(rule=terminal)

    def terminal_cap(m):
        return m.R <= 30
    model.c11 = am.Constraint(rule=terminal_cap)

    # ==========================================
    # SOLVE
    # ==========================================

    solver = am.SolverFactory('gurobi')
    solver.options['TimeLimit'] = 100
    solver.options['MIPGap'] = 0.06

    result = solver.solve(model, tee=True)

    # --- Extract bounds safely ---
    try:
        ub = float(result.problem.upper_bound)
    except:
        ub = None

    try:
        lb = float(result.problem.lower_bound)
    except:
        lb = None

    status = result.solver.termination_condition

    totalOrders=0.0
    TotalWaste=0.0
    UnmetDemand =0.0
    Leftoverinventory=0.0
    # --- helper to extract solution ---
    def extract_solution():
        totalOrders=0.0
        TotalWaste=0.0
        UnmetDemand =0.0
        output=[]
        for t in model.T:
            date=T_MAP[t]
            for i in model.S:
                totalOrders=totalOrders+float(model.Q[i,t].value or 0)
                TotalWaste=TotalWaste+float(model.W[i,t].value or 0)
                UnmetDemand=UnmetDemand + float(model.S_var[i,t].value or 0)
                output.append({
                    "date": date,
                    "item": i,
                    "order": float(model.Q[i,t].value or 0),
                    "shortage": float(model.S_var[i,t].value or 0),
                    "wastage": float(model.W[i,t].value or 0),
                    "inventory_fresh": float(model.I[i,1,t].value or 0)
                })
        result = {
            "output":output,
            "totalOrders":totalOrders,
            "totalWaste":TotalWaste,
            "unmetDemand":UnmetDemand
        }
        return result

    # ==========================================
    # RETURN JSON RESULT
    # ==========================================

    if status == am.TerminationCondition.optimal:
        result = extract_solution()
        print(type(result))
        return {
            "status": "optimal",
            "objective": float(am.value(model.obj)),
            "upper_bound": ub,
            "lower_bound": lb,
            "results": result["output"],
            "TotalOrders":result["totalOrders"],
            "TotalWaste":result["totalWaste"],
            "totalUnmetDemand":result["unmetDemand"],
        }

    elif status in [
        am.TerminationCondition.maxTimeLimit,
        am.TerminationCondition.feasible,
        am.TerminationCondition.other]:

        approx = (ub + lb)/2 if ub is not None and lb is not None else None
        result = extract_solution()
        print(type(result))
        return {
            "status": "feasible_not_optimal",
            "objective_best": float(am.value(model.obj)),
            "upper_bound": ub,
            "lower_bound": lb,
            "approx_objective_midpoint": approx,
            "results": result["output"],
            "TotalOrders":result["totalOrders"],
            "TotalWaste":result["totalWaste"],
            "totalUnmetDemand":result["unmetDemand"],
        }

    else:
        return {
            "status": "no_solution",
            "termination": str(status),
            "upper_bound": ub,
            "lower_bound": lb
        }
