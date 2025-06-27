# calculates the sum of squared residuals error as described in Millard's paper (objective function)
def ssr_error(standard_deviations_dict, observables, variable_data, variable_res):
    ssr = 0
    for obs in observables:
        for i in range(len(variable_data[obs])):
            ssr += ((variable_data[obs][i] - variable_res[obs][i])/standard_deviations_dict[obs])**2
    return ssr