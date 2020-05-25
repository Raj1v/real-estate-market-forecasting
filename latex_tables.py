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


class table_printer_2:

    def __init__(self, var_neg, var_pos, housing_var, housing_label, sent_lab_pos,
                 sent_lab_neg):
        self.var_results_pos = var_pos
        self.var_results_neg = var_neg
        self.housing_var = housing_var
        self.sent_var_pos = "Positive Scores"
        self.sent_var_neg = "Negative Scores"
        self.housing_label = housing_label
        self.sent_lab_pos = sent_lab_pos
        self.sent_lab_neg = sent_lab_neg
        self.k_pos = var_pos.k_ar
        self.k_neg = var_neg.k_ar
        pass

    def print_table(self):
        # const
        self.print_row('const', "Constant")

        # Housing lags
        self.print_lagged_parameters(self.housing_var, self.housing_label, max(self.k_pos, self.k_neg))

        # positive sentiment lags
        self.print_lagged_parameters(self.sent_var_pos, self.sent_lab_pos, self.k_pos)

        # negative sentiment lags
        self.print_lagged_parameters(self.sent_var_neg, self.sent_lab_neg, self.k_neg)

    def print_lagged_parameters(self, variable, label, k):
        housing_variable = self.housing_var

        for lag in range(1, k + 1):
            row_variable = "L{lag}.{variable}".format(lag=lag, variable=variable)
            row_label = "{label} \\textit{{(-{lag})}}".format(label=label, lag=lag)
            self.print_row(row_variable, row_label)

    def print_row(self, variable, label):
        print(label)
        for var_results in [self.var_results_pos, self.var_results_neg]:
            self.print_row_part(variable, var_results)
        print("\\\\")

    def print_row_part(self, variable, var_results):
        housing_variable = self.housing_var
        if variable not in var_results.params[housing_variable]:
            print("& & & & &")
            return
        coef = var_results.params[housing_variable][variable]
        coef = "{:.3e}".format(coef)
        se = var_results.stderr[housing_variable][variable]
        se = "{:.3e}".format(se)
        pvalue = var_results.pvalues[housing_variable][variable]
        pvalue = round(pvalue, 3)
        tstat = var_results.tvalues[housing_variable][variable]
        tstat = round(tstat, 3)
        significant = is_significant(pvalue)
        lower_bound, upper_bound = self.confidence_interval(variable, var_results)
        lower_bound, upper_bound = "{:.3e}".format(lower_bound), "{:.3e}".format(upper_bound)

        row = "& \\num{{{coef}}} & \\num{{{se}}} & ${pvalue}$ {signif} & \\num{{{lb}}} & \\num{{{" \
              "ub}}}".format(
            coef=coef, se=se,
            tstat=tstat,
            pvalue=pvalue,
            signif=significant,
            lb=lower_bound,
            ub=upper_bound)
        print(row)

    def confidence_interval(self, variable, var_results):
        housing_variable = self.housing_var
        n = var_results.nobs
        coefficient = var_results.params[housing_variable][variable]
        se = var_results.stderr[housing_variable][variable]
        h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
        lower_bound = coefficient - h
        upper_bound = coefficient + h
        return lower_bound, upper_bound
