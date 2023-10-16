#!/usr/bin/env python
# coding: utf-8

# # STAFFING PLANNING OPTIMIZATION - CASE STUDY

# An insurance company wants you to help them with finding the optimal number of staff that they need for their insurance application approval process. In the industry, the number of staff is considered a continuous variable. This is also called a Full-Time Equivalent (FTE) of the staff. 
# 
# The company can either handle an application with the staff that they hire or outsource it to a vendor. Assume that there is no capacity limitation to outsourcing.
# 
# If they hire staff, he/she can handle 40 insurance applications per month when he/she works 100% of the workdays. However, there are days that he/she will be unavailable to process applications due to training, off days, etc.
# 
# States A and B have a regulatory restriction that the outsourced insurance applications cannot be more than 30% and 40% of the total number of applications for each month, respectively.
# 
# The objective is to optimise the total cost for the application approval process by distributing the right number of applications between the FTEs and the vendors while meeting the monthly demand for each state at the same time.
# 
# Note: This program focuses on the analysis of Staffing Optimization when employees works at the averafe FTE.

# In[1]:


import pandas as pd
import numpy as np
import math
from pyomo.environ import *

import warnings
warnings.filterwarnings('ignore')


# In[2]:


from __future__ import division
from pyomo.opt import SolverFactory


# In[18]:


# Reading the data from Excel workbook - DemandData
demand = pd.read_excel("/Users/rolanddelarosa/Desktop/StaffInsurance.xlsx",sheet_name='DemandData')

# Reading the data from Excel workbook - StaffAvailability
staff_data = pd.read_excel("/Users/rolanddelarosa/Desktop/StaffInsurance.xlsx",sheet_name='StaffAvailability')

# Reading the data from Excel workbook - Cost
cost = pd.read_excel("/Users/rolanddelarosa/Desktop/StaffInsurance.xlsx",sheet_name='Cost')

# Number of insurance applications that can be processed by an FTE in a month when working with 100% availability
srv_rate = 40


# In[20]:


# Viewing the exported data
demand.head()


# In[21]:


# Viewing the exported data
staff_data.head()


# In[22]:


# Viewing the exported data
cost.head()


# Data Pre-processing

# In[25]:


# Creating unique index values from the dataset
month_name = demand['Month'].unique()
state_name = demand['State'].unique()
print("state :",state_name)
print("months :",month_name)


# In[26]:


# Creating dict for the required parameters with [State, Month] as indexes

#Application Demand across State & Month
Demand_data = demand.set_index(['State','Month'])['Demand'].to_dict()

# Cost per Application for Outsource 
OutsourceRate = cost.set_index(['State','Month'])['UnitOutSourceCost'].to_dict()

# Cost for FTE monthly salary 
StaffMsal = cost.set_index(['State','Month'])['MonthlySalary'].to_dict()

# Staff availabilty for serving the Insurance - regular scenario , Lower bound , Upper bound scenario
StaffAvPer_data = staff_data.set_index(['State','Month'])['StaffAvPer'].to_dict()
StaffLB_data = staff_data.set_index(['State','Month'])['LB'].to_dict()
StaffUB_data = staff_data.set_index(['State','Month'])['UB'].to_dict()

Staff_app_permonth = 40


# Question 1
# 
# The company wants to know the optimised staffing recommendations for the business case described. Write the mathematical model for the deterministic optimisation problem. Define and explain your decision variables, objective function and the constraint. (Hint: Use months of the year as the model timeline).

# Answer to Question 1

# Objective: The objective of the case study is to identify the optimal number of staff between Employees & Outsource such as to minimize the total cost for application approval process

# Case Study Objective :
# 
# The objective of the case study is to identify the optimal number of staff between Employees & Outsource such as to minimize the total cost for application approval process
# 
# Decision Variables :
# 
# The objective indicates towards requirement of Cost & Count for developing the mathematical equation of the Objective Function. For Outsource - the cost is provided at per application basis & hence the 1st Variable  will be No. of Applications from Outsource named - Outsource_appl across states & months For Employess - the cost is provided as monthly salary & hence the  will be No. of FTE named - FTE_count across states & months
# 
# Objective Function :
# 
# To minimize the total cost for application approval process the Objective function will be :
# 
# x * 40 * a + y = D
# 
# where,
# 
# S = States
# M = Months
# c = Staff monthly salary
# x = no. of FTE
# O = Outsource Cost
# y = No. of applications by outsource
# a = Staff Availability %
# D = Application Demand
# 40 = No. of Application per month by staff incase of 100% availability
# Constraints
# 
# 1. Demand per month per state
# 
# Before we create the Demand Constraint, we need to calculate a variable for No. of Applications being processed by FTEs.
# - FTE_Appl = No. of FTE * StaffAvPer_data * Staff_app_permonth Demand_Constraint = FTE_Appl + Outsource_appl for each month for each state
# 
# 
# 2. Maximum Outsource for State A
# 
# State A can only provide 30% of the total application demand to outsource
# 
# 3. Maximum Outsource for State B
# 
# State B can only provide 40% of the total application demand to outsource

# Question 2
# 
# Code the problem is Python and use any optimization package to solve it. Add comments to your code to explain each step.
# 
# Expected output:
# 
# Create a data frame containing the number of outsourced applications and the number of FTEs for each state-month combination. You can choose to have extra columns like staff availability, demand etc. in your dataframe apart from the ones mentioned earlier.

# In[27]:


# Creating a model instance
model = ConcreteModel()


# Pyomo sets and Parameters

# In[28]:


# Initializing month & state for usage in loops ahead
model.location = Set(initialize = state_name.tolist(),doc = 'States')
model.months = Set(initialize=month_name.tolist(), doc = 'Months')


# In[29]:


model.location.display()


# In[30]:


# Defining model Parameter for Demand across months & states
model.d = Param(model.location, model.months, initialize = Demand_data, doc = 'Demand')
model.d.display()


# In[31]:


# Defining model Parameter for Staff Availability across months & states
model.avail = Param(model.location, model.months, initialize = StaffAvPer_data, doc = 'Staff_Availability')
model.avail.display()


# In[32]:


# Defining model Parameter for Staff Monthly Salary across months & states
model.staffsalary = Param(model.location,model.months, initialize=StaffMsal, doc='Staff Monthly Salary')
model.staffsalary.display()


# In[33]:


# Defining model Parameter for Outsource Cost across months & states
model.outsourcecost = Param(model.location, model.months, initialize = OutsourceRate, doc = 'Outsource Cost')
model.outsourcecost.display()


# Decision Variables

# In[34]:


# Decision variables
model.x = Var(model.location, model.months, domain=NonNegativeReals, doc='No. of FTE')
model.y = Var(model.location, model.months, domain=NonNegativeIntegers, doc='No. of Outsource Appl.')


# Objective Function

# In[35]:


obj = 0
def obj_rule(model):
    global obj
    for l in model.location:
        for m in model.months:
            # formula used is : (Outsource Appl * per Appl cost for Outsource) + (no. of FTEs * Monthly Salary)
            obj+=(model.y[l,m]*model.outsourcecost[l,m])+(model.x[l,m]*model.staffsalary[l,m])
    return obj
model.output = Objective(rule=obj_rule, sense= minimize)


# Constraints - Demand Constraint

# In[36]:


model.ddConst = ConstraintList()
for l in model.location:
    for m in model.months:
        model.ddConst.add(expr = model.y[l,m] + model.x[l,m]*(model.avail[l,m]*40) == model.d[l,m]) 
        # Demand = No. of Outsource Applications + (FTE count* Max Application,i.e.,'40' * Staff Availability %)


# Constraints - State A Outsourcing Constraint

# In[37]:


model.statelimit = ConstraintList()

for l in model.location:
    for m in model.months:
        if l == 'A':
            model.statelimit.add(expr = model.y[l,m] <= 0.3*model.d[l,m])
        elif l == 'B':
            model.statelimit.add(expr = model.y[l,m] <= 0.4*model.d[l,m])
        else:
            pass


# Invoking Solver

# In[42]:


from pyomo.opt import SolverFactory

# Specify the full path to the glpsol executable
solver_path = '/opt/homebrew/bin/glpsol'  # Update this path if it's different on your system

# Invoking the solver
result = SolverFactory('glpk', executable=solver_path).solve(model)
result.write()


# In[43]:


# Print the value of the objective function (converting the output to million)
base_cost = int(model.output())/10**6
base_cost


# The company has to spend approzimately around 17.9 m$ in total for the application approval process.

# In[44]:


# Creating dataframe for the results
# optimal number of staff members for the Avg case scenario
Avg_Output = []
Avg_avilb = 0

for l in model.location:
    for m in model.months:
        
        # optimal no. of staff members for Avg Case scenario
        Staff_count = (round(model.x[l,m].value))
        
        # outsourced applications% = no. of Outsource applications(stored in variable y) divide by total demand(stored in variable d)*100
        Outsource_applications = (model.y[l,m].value)
        
        # Total Cost = cost of staff + Cost of   outsourced  applications
                
        Avg_Output.append([l,m, Staff_count, Outsource_applications])
       
                
print(Avg_Output)


# In[45]:


Avg_Output = pd.DataFrame(Avg_Output, columns=['State','Month','Optimal Staff_AvgCase','Out_Appl_AvgCase'])
Avg_Output


# In[46]:


# Importing the required library
from openpyxl import load_workbook


# In[48]:


# Writing the results in to an Excel sheet
# Writing the output data into an Excel sheet named "Staffing_Data.xlsx" workbook


book = load_workbook("/Users/rolanddelarosa/Desktop/StaffInsurance.xlsx")

# create excel writer object
writer = pd.ExcelWriter("/Users/rolanddelarosa/Desktop/StaffInsurance.xlsx", engine = 'openpyxl')


#Assigning the workbook to the writer object
writer.book = book


# write dataframe to excel sheet named 'output'
Avg_Output.to_excel(writer, sheet_name='Output_Staff_Planning_Avg')



# save the excel file
writer.save()
print('DataFrame is written successfully to Excel Sheet.')


# In[49]:


# creating dataframe to record the Overall output of Average Scenario for further scenario

from statistics import mean
final_output = []
appl_cost = 0
out_appl = 0
for l in model.location:
    for m in model.months:
            
            cost_per_appl=((model.y[l,m].value*model.outsourcecost[l,m])+(model.x[l,m].value*model.staffsalary[l,m]))/model.d[l,m]
            appl_cost+=cost_per_appl
            final_output.append(cost_per_appl)
            out_appl+= model.y[l,m].value
            
base_mean = mean(final_output)
overall_df = pd.DataFrame(data={(round(base_mean,1),round(appl_cost,1),round(base_cost,2),round(out_appl,))},index=['Base Scenario'],columns=['Avg_cost_per_appl','tot_cost_per_appl','Overall Cost_in $mio','Outsource_Appl_count'])
overall_df

