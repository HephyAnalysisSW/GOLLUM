from iminuit import Minuit

def testFunction_onlyMu(mu):
    return (mu-1.5)*(mu-1.5)

def testFunction(mu, nu1, nu2):
    return (mu - nu1+1)**2 + (mu + nu2-1)**2 + mu * mu



################################################################################
print("========================================")
print("1. Function with only one param")
errordef = Minuit.LEAST_SQUARES
m = Minuit(testFunction_onlyMu, mu=0.0)
m.migrad()
print("Minimum value:", m.fval)
print("Best parameters:", m.values)
################################################################################
print("========================================")
print("2. Function with mu, nu1, nu2: find global min")
errordef = Minuit.LEAST_SQUARES
m = Minuit(testFunction, mu=0.0, nu1=0.1, nu2=0.0)
m.migrad()
print("Minimum value:", m.fval)
print("Best parameters:", m.values)
################################################################################
print("========================================")
print("3. Function with mu, nu1, nu2: scan over mu")
Nsteps = 20
mumin = -5
mumax = 5
muList = [mumin+i*(mumax-mumin)/Nsteps for i in range(Nsteps)]
errordef = Minuit.LEAST_SQUARES
for mu in muList:
    fixed_mu = mu
    fixed_function = lambda nu1, nu2: testFunction(fixed_mu, nu1, nu2)
    m = Minuit(fixed_function, nu1=0.1, nu2=0.0)
    m.migrad()
    print("-------------")
    print(f"mu = {mu}")
    print("Minimum value:", m.fval)
    print("Best parameters:", m.values)
