import scipy


class table_printer:

    def __init__(self, var_results, housing_variable, sentiment_variable, housing_label, sentiment_label):
        self.var_results = var_results
        self.housing_variable = housing_variable
        self.sentiment_variable = sentiment_variable
        self.housing_label = housing_label
        self.sentiment_label = sentiment_label
        self.k = var_results.k_ar
        pass

    def print_table(self):
        # const
        self.print_row('const', "Constant")

        # Housing lags
        self.print_lagged_parameters(self.housing_variable, self.housing_label)

        # sentiment lags
        self.print_lagged_parameters(self.sentiment_variable, self.sentiment_label)

    def print_lagged_parameters(self, variable, label):
        k = self.k
        var_results = self.var_results
        housing_variable = self.housing_variable

        for lag in range(1, k + 1):
            row_variable = "L{lag}.{variable}".format(lag=lag, variable=variable)
            row_label = "{label} \\textit{{(-{lag})}}".format(label=label, lag=lag)
            self.print_row(row_variable, row_label)

    def print_row(self, variable, label):
        var_results = self.var_results
        housing_variable = self.housing_variable
        coef = var_results.params[housing_variable][variable]
        coef = "{:.3e}".format(coef)
        se = var_results.stderr[housing_variable][variable]
        se = "{:.3e}".format(se)
        pvalue = var_results.pvalues[housing_variable][variable]
        pvalue = round(pvalue, 3)
        tstat = var_results.tvalues[housing_variable][variable]
        tstat = round(tstat, 3)
        significant = is_significant(pvalue)
        lower_bound, upper_bound = self.confidence_interval(variable)
        lower_bound, upper_bound = "{:.3e}".format(lower_bound), "{:.3e}".format(upper_bound)

        row = "{label} & \\num{{{coef}}} & \\num{{{se}}} & ${tstat}$ & ${pvalue}$ {signif} & \\num{{{lb}}} & \\num{{{" \
              "ub}}} \\\\".format(
            label=label,
            coef=coef, se=se,
            tstat=tstat,
            pvalue=pvalue,
            signif=significant,
            lb=lower_bound,
            ub=upper_bound)
        print(row)

    def confidence_interval(self, variable):
        var_results = self.var_results
        housing_variable = self.housing_variable
        n = var_results.nobs
        coefficient = var_results.params[housing_variable][variable]
        se = var_results.stderr[housing_variable][variable]
        h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
        lower_bound = coefficient - h
        upper_bound = coefficient + h
        return lower_bound, upper_bound


def is_significant(p_value):
    if p_value < 0.05:
        return "\\tnote{*}"
    else:
        return ""

