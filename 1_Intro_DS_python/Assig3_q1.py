def answer_one():
    energy = pd.read_excel('Energy Indicators.xls',skipfooter=38, encoding='ANSI', header = 9)
    energy = energy.iloc[8:,-4:]
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy.index = list(range(227))
    energy[ 'Energy Supply'] = energy['Energy Supply'].apply(lambda x :np.NaN if x == '...' else x)
    energy['Energy Supply per Capita'] = energy['Energy Supply per Capita'].apply(lambda x :np.NaN if x == '...' else x)
    energy['Energy Supply'] = energy['Energy Supply']*1000000
    def t(x):
        if x[-2].isdigit():
            return x[:-2]
        if x[-1].isdigit():
            return x[:-1]
        if x[-1] == ")" :
            index = x.index("(")
            return x[:index-1]
        return x
    energy['Country'] = energy['Country'].apply(t)
    energy['Country'].replace({ "Republic of Korea" : "South Korea" ,
      "United States of America": "United States",
      "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
      "China, Hong Kong Special Administrative Region": "Hong Kong"},inplace = True)
    GDP = pd.read_csv("world_bank.csv",header=4)
    GDP['Country Name'].replace({"Korea, Rep.": "South Korea",
       "Iran, Islamic Rep.": "Iran",
       "Hong Kong SAR, China": "Hong Kong"},inplace=True)
    ScimEn = pd.read_excel("scimagojr-3.xlsx",header=0)
    GDP.rename(columns={'Country Name':'Country'},inplace=True)
    energy.set_index('Country',inplace=True)
    GDP.set_index('Country',inplace=True)
    ScimEn.set_index('Country',inplace=True)

    df = pd.merge(ScimEn.iloc[:15,:],energy, how = 'inner', left_index=True , right_index=True)
    df = pd.merge(df,GDP.iloc[:,-10:], how = 'inner', left_index=True , right_index=True)

    return df