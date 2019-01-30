import statsmodels.regression.linear_model as sm
import pandas as pd

def Stepwise_Forward_Selection(Data, Inputs,Output):
            Model_var1=sm.OLS
            X=Data[Inputs]
            y=Data[Output]
            initial_list=[]
            threshold_in=0.05
            verbose=True
            included = list(initial_list)
            while True:
                changed=False
                excluded = list(set(X.columns)-set(included))
                new_pval = pd.Series(index=excluded)
                for new_column in excluded:
                    model = Model_var1(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
                    new_pval[new_column] = model.pvalues[new_column]
                best_pval = new_pval.min()
                if best_pval < threshold_in:
                    best_feature = new_pval.argmin()
                    included.append(best_feature)
                    changed=True
                    if verbose:
                        print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
                if not changed:
                    break
            return included

def build_csv(data, columnlist):
    newdata = pd.DataFrame()
    for i in columnlist:
        newdata[i] = data[i]

    newdata.to_csv("Stepwise_selected.csv", index=False)

def use_package():
    df = pd.read_csv("./14_input_data.csv")

    included = Stepwise_Forward_Selection(df,list(df.columns)[:-1],['SalePrice'])

    build_csv(df, included)

use_package()