"""Extra randomized checks for weighted_node_costs.py (larger trees; min-cost FPTAS)."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weighted_node_costs import *

def big():
    rng=random.Random(7); ok=[0,0,0,0]; strict=0; N=600
    for t in range(N):
        n=rng.randint(8,15); ch,cost=random_instance(n,rng,cmax=10,wmax=5,dmax=12)
        I=internal(ch); B=rng.randint(0,sum(cost[v] for v in I))
        opt,_=brute_max(ch,cost,B); ok[0]+=dp_max_budget(ch,cost,B)[B]==opt
        Tm=value(ch,set(I)); T=rng.randint(0,Tm+1)
        ok[1]+=dp_min_cost_table(ch,cost,Tm+2)[T]==brute_mincost(ch,cost,T)
        eps=rng.choice([0.5,0.3]); S=fptas_max(ch,cost,B,eps); v=value(ch,S)
        ok[2]+= sum(cost[x] for x in S)<=B and v>=(1-eps)*opt-1e-9; strict+= v<opt
        # unit costs: DP equals brute cardinality version
        c1=[1]*n; K=rng.randint(0,len(I)); ok[3]+=dp_max_budget(ch,c1,K)[K]==brute_max(ch,c1,K)[0]
    print('trials',N,'maxDP',ok[0],'mincostDP',ok[1],'fptas_max',ok[2],'(strictly suboptimal',strict,')','unitK',ok[3])

def fmin():
    rng=random.Random(99); ok=chk=strict=0
    for t in range(500):
        n=rng.randint(4,10); ch,cost=random_instance(n,rng,cmax=12)
        I=internal(ch); Tm=value(ch,set(I)); T=rng.randint(0,Tm+1); eps=rng.choice([0.5,0.2])
        bc=brute_mincost(ch,cost,T); S=fptas_mincost(ch,cost,T,eps); chk+=1
        if S is None: ok+= bc==INF; continue
        cs=sum(cost[v] for v in S); ok+= value(ch,S)>=T and cs<=(1+eps)*bc+1e-9; strict+= cs>bc
    print('fptas_min checked',chk,'ok',ok,'strictly costlier than OPT',strict)

if __name__ == '__main__':
    big()
    fmin()
