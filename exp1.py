from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

model = BayesianModel([('Age', 'HeartDisease'), ('FamilyHistory', 'HeartDisease')])

cpd_age = TabularCPD(variable='Age', variable_card=3, values=[[0.2], [0.4], [0.4]])
cpd_family_history = TabularCPD(variable='FamilyHistory', variable_card=2, values=[[0.7], [0.3]])
cpd_heart_disease = TabularCPD(variable='HeartDisease', variable_card=2,
                               values=[[0.95, 0.6, 0.9, 0.3,0.5,0.1],
                                       [0.05, 0.4, 0.1, 0.7,0.5,0.9]],
                               evidence=['Age', 'FamilyHistory'],
                               evidence_card=[3, 2])

model.add_cpds(cpd_age, cpd_family_history, cpd_heart_disease)

model.check_model()

from pgmpy.inference import VariableElimination

infer = VariableElimination(model)
evidence = {"Age":0,"FamilyHistory":1}
query = infer.query(["HeartDisease"],evidence=evidence)
print(query)
