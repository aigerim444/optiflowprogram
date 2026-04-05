import gurobipy as gp
from gurobipy import GRB

# -------------------------------------------------------
# INPUTS
# -------------------------------------------------------

S   = set()          # set of all segment indices
A   = set()          # set of (i,j) tuples: i flows into j
S1  = set()          # segments that have an upstream predecessor
S2  = set()          # segments with no predecessor (S \ S1)

r   = {}             # r[i] = rainfall catchment volume for segment i
c   = {}             # c[i] = capacity of segment i (with dam height)
h   = {}             # h[i] = habitat heuristic value in [0,1]

Lu  = {}             # Lu[i] = set of segments uphill of i to same inlet (incl. i)
Ld  = {}             # Ld[i] = set of segments downhill of i to same inlet (incl. i)
u   = {}             # u[i] = the unique j in Lu[i] ∩ S2

Td  = 10             # max dams to install
Tf  = 10             # max forebays to install
a   = 1.0            # weight for water capture in objective
b   = 0.5            # weight for habitat preservation in objective

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------

model = gp.Model("OptiFlow")

# --- Decision Variables ---
x = model.addVars(S, vtype=GRB.BINARY, name="x")           # dam at downhill side of i
y = model.addVars(S, vtype=GRB.BINARY, name="y")           # forebay at uphill side of i
z = model.addVars(S, vtype=GRB.CONTINUOUS, name="z", lb=0) # water flow into segment i
v = model.addVars(S, vtype=GRB.CONTINUOUS, name="v", lb=0) # water captured/treated by i

# --- Objective ---
# Maximize treated water - penalty for placing dams in high-habitat areas
model.setObjective(
    a * gp.quicksum(v[i] for i in S)
    - b * gp.quicksum((1 - h[i]) * x[i] for i in S),
    GRB.MAXIMIZE
)

# --- Constraints ---

# Limit total dams and forebays
model.addConstr(gp.quicksum(x[i] for (i,j) in A) <= Td, "max_dams")
model.addConstr(gp.quicksum(y[i] for (i,j) in A) <= Tf, "max_forebays")

# Flow balance for S1 segments (have upstream predecessor)
for (i, j) in A:
    if j in S1:
        model.addConstr(z[j] == r[j] + z[i] - v[i], f"flow_S1_{j}")

# Flow initialization for S2 segments (no predecessor, just rainfall)
for i in S2:
    model.addConstr(z[i] == r[i], f"flow_S2_{i}")

# Treated water can't exceed what flows in
for i in S:
    model.addConstr(v[i] <= z[i], f"treat_cap_flow_{i}")

# Treated water can't exceed capacity — needs dam below (in Ld)
for i in S:
    model.addConstr(
        v[i] <= c[i] * gp.quicksum(y[j] for j in Ld[i]),
        f"treat_cap_dam_below_{i}"
    )

# Treated water can't exceed capacity — needs forebay above (in Lu)
for i in S:
    model.addConstr(
        v[i] <= c[i] * gp.quicksum(y[j] for j in Lu[i]),
        f"treat_cap_forebay_above_{i}"
    )

# Absolute capacity ceiling
for i in S:
    model.addConstr(v[i] <= c[i], f"absolute_cap_{i}")

# --- Solve ---
model.optimize()

# --- Results ---
if model.status == GRB.OPTIMAL:
    print(f"Objective: {model.objVal:.4f}")
    dams    = [i for i in S if x[i].X > 0.5]
    forebays = [i for i in S if y[i].X > 0.5]
    print(f"Dam locations:     {dams}")
    print(f"Forebay locations: {forebays}")
    print(f"Total water treated: {sum(v[i].X for i in S):.2f}")