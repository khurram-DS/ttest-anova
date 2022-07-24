import streamlit as st
# Eda packages

import pandas as pd
import numpy as np

#Data viz packages

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

#function

def main():
    
    title_container1 = st.container()
    col1, col2 ,  = st.columns([6,12])
    from PIL import Image
    image = Image.open('static/asia.jpeg')
    with title_container1:
        with col1:
            st.image(image, width=200)
        with col2:
            st.markdown('<h1 style="color: tomato;">ASIA Consulting</h1>',
                           unsafe_allow_html=True)
    
    
    
    
    
    st.sidebar.image("static/rhe.jpg", use_column_width=True)
    activites = ["About","t-test & ANOVA"]
    choice =st.sidebar.selectbox("Select Activity",activites)
 
    if choice == 'About':
        st.subheader("About t-test & ANOVA")
        title_container1 = st.container()
        col1, col2 ,  = st.columns([4,20])
        from PIL import Image
        image = Image.open('static/Logo.jpg')
        with title_container1:
            
            with col2:
                st.image(image, width=400, caption='t-test & ANOVA')
            
        st.markdown("""**❓ What is t-test & ANOVA?**
                    
The t-test is a method that determines whether two populations are statistically different from each other, whereas ANOVA determines whether three or more populations are statistically different from each other.""")
        
        title_container1 = st.container()
        col1, col2 ,  = st.columns([12,12])
        from PIL import Image
        image = Image.open('static/rheu.png')
        with title_container1:
            with col1:
                st.image(image, width=340, caption='t-test')
                
            with col2:
                st.image("https://dacg.in/wp-content/uploads/2020/03/anova.jpg", width=430, caption='ANOVA test')

        
        
        
        

    elif choice == 't-test & ANOVA':
        
        st.subheader(" t-test & ANOVA test")
        
        
        

       
        @st.cache(allow_output_mutation=True)
        def get_df(file):
          # get extension and read file
          extension = file.name.split('.')[1]
          if extension.upper() == 'CSV':
            df = pd.read_csv(file)
          elif extension.upper() == 'XLSX':
            df = pd.read_excel(file)
          
          return df
        file = st.file_uploader("Upload file", type=['csv' 
                                                 ,'xlsx'])
        if not file:
            st.write("Upload a .csv or .xlsx file to get started")
            return
          
        df = get_df(file)
        if st.checkbox("Show Raw Data"):
          
            st.write(df.head())
            
        if st.checkbox("Show summary statistics for Overall Data"):
            st.subheader('Summary statistics')
            import numpy as np
            from scipy.stats import kurtosis, skew
            stats=df.describe()
            stats.loc['SEM']=df.sem().tolist()
            stats.loc['skewness']=df.skew().tolist()
            stats.loc['kurtosis']=df.kurtosis().tolist()
            st.write(stats.T)   
        if st.checkbox("Select Column to see frequency table"):
            all_columns=df.columns.to_list()
            selected_columns= st.selectbox("Select Columns to see Counts and frequency", all_columns)
            cnt=df[selected_columns].value_counts()
            per=df[selected_columns].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
            
            Course=pd.DataFrame({'counts': cnt,'Frequency %': per})
            Course.reset_index(inplace = True)
            Course.rename(columns={'index':'Course'},inplace=True)
            Course['Total_data']=Course['counts'].sum()
            st.dataframe(Course)
        
        st.markdown("<h1 style='text-align: center; color: black;'>Check Results for test </h1>", unsafe_allow_html=True)
        
        
        st.markdown("**Select 1st columns as a categorical column to perform tests**")
        all_columns1=df.columns.to_list()
        selected_columns1= st.selectbox("Select categorical column", all_columns1)
        #df[selected_columns].value_counts()
        
        
        all_columns2=df.columns.to_list()
        st.markdown("**Select 2nd columns as a Numeric column to perform tests**")
        selected_columns2= st.selectbox("Select Numeric column", all_columns2)
        
        length=len(df[selected_columns1].value_counts())
        typ=df[selected_columns1].unique()
        if length == 2:
            
            if st.checkbox("click to see the Results for Selected Columns"):
                st.subheader("Two-Tailed Independent Samples t-Test")
                
                st.markdown("**Introduction**")
                
                st.write("""A two-tailed independent samples t-test was conducted to examine whether the mean of {} was significantly different between the {} and {} categories of {}""".format(selected_columns2,typ[0],typ[1],selected_columns1))            
                st.markdown("**Results**")
                
                from pylab import rcParams
                from scipy.stats import f_oneway
                from scipy.stats import ttest_ind
                import seaborn as sns
                import numpy as np
                import seaborn as sns

                import plotly.figure_factory as ff
                import numpy as np
                from plotly.offline import init_notebook_mode, iplot
                import plotly.figure_factory as ff
                import cufflinks
                cufflinks.go_offline()
                cufflinks.set_config_file(world_readable=True, theme='pearl')
                import plotly.graph_objs as go
                from chart_studio import plotly as py
                import plotly
                from plotly import tools
                
                import warnings            
                warnings.filterwarnings("ignore")
                Gender_a=df.loc[df[selected_columns1] == typ[0]][selected_columns2]
                Gender_b=df.loc[df[selected_columns1] == typ[1]][selected_columns2]            
                rcParams['figure.figsize'] = 20,10
                rcParams['font.size'] = 30
                sns.set()
                np.random.seed(8)
                
                
                
                import pingouin as pg
    
                st.write(pg.ttest(Gender_a, Gender_b, correction=False))
                tres=pg.ttest(Gender_a, Gender_b, correction=False)
                
                
                stat=df.groupby(selected_columns1)[selected_columns2].describe()
                
                stat['skewness']=df.groupby(selected_columns1)[selected_columns2].skew()
                stat['SEM']=df.groupby(selected_columns1)[selected_columns2].sem()
                stat['N']=df.groupby(selected_columns1)[selected_columns2].value_counts().sum()
                st.write(stat)
               
                
                
                st.write("""- The result of the two-tailed independent samples t-test was significant based on an alpha value of .05, t({}) = {}, p = {}""".format(tres['dof'][0],tres['T'][0],tres['p-val'][0])) 
                alpha=0.05
                if tres['p-val'][0] > alpha:
                    st.write('✍  As p-value is Greater than alpha=0.05 thus, Sample looks Gaussian (fail to reject H0)')
                else:
                    st.write('✍  As p-value is lesser than alpha=0.05 thus,Sample does not look Gaussian (reject H0)')
                    
                st.text('● The p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test')
                st.text('➊ Null hypotheses(HO): Two group means are equal')
                st.text('➋ Alternative hypotheses(H1): Two group means are different')
                
                st.write("The results are presented in Table, and the plot you can see in the deeper part")
                
                
                
                st.subheader("Lets understand the t-test result more deeply for {} and {}".format(selected_columns1,selected_columns2)) 
                
                if st.checkbox("Show the t-test for {} | {}-{}. - {} | {}-{}".format(selected_columns2,selected_columns1,typ[0],selected_columns2,selected_columns1,typ[1])):
                    st.subheader('Making some Asuumations')
                    st.text('Assumption 1: Are the two samples independent?')
                    st.text('Assumption 2: Are the data from each of the 2 groups following a normal distribution?')
                    
                    
                    st.subheader('Checking the Normality of Data')
                    st.text(" Checking normality of data for {} using shapiro test".format(typ[0]))
                    
                    from scipy.stats import shapiro
                    stat, p = shapiro(Gender_a)
                    
                    # interpret
                    alpha = 0.05
                    if p > alpha:
                        msg = 'Sample looks Gaussian (fail to reject H0)'
                    else:
                        msg = 'Sample does not look Gaussian (reject H0)'
                    
                    result_mat = [
                        ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                        [len(Gender_a), stat, p, msg]
                    ]
                    
                    import plotly.figure_factory as ff
                    swt_table = ff.create_table(result_mat)
                    swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                    swt_table['layout']['height']=200
                    swt_table['layout']['margin']['t']=50
                    swt_table['layout']['margin']['b']=50
                    
                    #py.iplot(swt_table, filename='shapiro-wilk-table')
                    st.write(swt_table)
                    
                    st.text('Checking normality of data for {} using shapiro test'.format(typ[1]))
                    
                    from scipy.stats import shapiro
                    stat, p = shapiro(Gender_b)
                    
                    # interpret
                    alpha = 0.05
                    if p > alpha:
                        msg = 'Sample looks Gaussian (fail to reject H0)'
                    else:
                        msg = 'Sample does not look Gaussian (reject H0)'
                    
                    result_mat = [
                        ['Length of the sample data', 'Test Statistic', 'p-value', 'Comments'],
                        [len(Gender_b), stat, p, msg]
                    ]
                    import plotly.figure_factory as ff
                    swt_table = ff.create_table(result_mat)
                    swt_table['data'][0].colorscale=[[0, '#2a3f5f'],[1, '#ffffff']]
                    swt_table['layout']['height']=200
                    swt_table['layout']['margin']['t']=50
                    swt_table['layout']['margin']['b']=50
                    
                    
                    st.write(swt_table)
                    import pingouin as pg
    
                    res = pg.ttest(Gender_a, Gender_b, correction=False)
                    st.subheader('Test interpretation: results ')
                    st.write('✍  T-value is  [[ {} ]] '.format(res['T'][0]))
                    st.text('● T is simply the calculated difference represented in units of standard error')
                    st.write('✍ Degree of freedom is [[ {} ]]'.format(res['dof'][0]))
                    st.text('● Degrees of freedom refers to the maximum number of logically independent values')
                    st.write('✍ 95% confidence interval on the difference between the means: is [[ {} ]]'.format(res['CI95%'][0]))
                    st.text('● A 95% CI simply means that if the study is conducted multiple times (multiple sampling from the same population)')
                    st.write('✍ Cohens d (realtive strength) value is [[ {} ]]'.format(res['CI95%'][0]))
                    st.text('● Cohens d is an effect size used to indicate the standardised difference between two means ')
                    
                    
                    alpha=0.05
                    if res['p-val'][0] > alpha:
                        st.write('✍  As p-value is Greater than alpha=0.05 thus, Sample looks Gaussian (fail to reject H0)')
                    else:
                        st.write('✍  As p-value is lesser than alpha=0.05 thus,Sample does not look Gaussian (reject H0)')
                        
                    st.text('● The p-value is the probability of obtaining results at least as extreme as the observed results of a statistical hypothesis test')
                    st.text('➊ Null hypotheses(HO): Two group means are equal')
                    st.text('➋ Alternative hypotheses(H1): Two group means are different')
                    
                    st.subheader("Plot for {} | {}-{}. - {} | {}-{}".format(selected_columns2,selected_columns1,typ[0],selected_columns2,selected_columns1,typ[1]))
                    import matplotlib.pyplot as plt
                    def plot_distribution(inp):
                        plt.figure()
                        ax = sns.distplot(inp)
                        plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
                        _, max_ = plt.ylim()
                        plt.text(
                            inp.mean() + inp.mean() / 10,
                            max_ - max_ / 10,
                            "Mean: {:.2f}".format(inp.mean()),
                        )
                        return plt.figure
                    
                    ax1 = sns.distplot(Gender_a)
                    ax2 = sns.distplot(Gender_b)
                    plt.axvline(np.mean(Gender_a), color='b', linestyle='dashed', linewidth=5)
                    plt.axvline(np.mean(Gender_b), color='orange', linestyle='dashed', linewidth=5)
                    st.pyplot(plt)
                    showPyplotGlobalUse = False
                    #boxplot
                    if st.checkbox("Show Plots for different {}, based on {}".format(selected_columns1,selected_columns2)):
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        from plotly.offline import init_notebook_mode, iplot
                        import plotly.figure_factory as ff
                        import cufflinks
                        cufflinks.go_offline()
                        cufflinks.set_config_file(world_readable=True, theme='pearl')
                        import plotly.graph_objs as go
                        import chart_studio.plotly as py
                        import plotly.offline as py
                        import plotly
                        from plotly import tools
                        trace0 = go.Box(
                                y=df.loc[df[selected_columns1] == typ[0]][selected_columns2],
                                name = typ[0],
                                marker = dict(
                                    color = 'rgb(214, 12, 140)',
                                )
                            )
                        trace1 = go.Box(
                                y=df.loc[df[selected_columns1] == typ[1]][selected_columns2],
                                name = typ[1],
                                marker = dict(
                                    color = 'rgb(0, 128, 128)',
                                )
                            )
                        
                            
                        data1 = [trace0, trace1]
                        layout1 = go.Layout(
                                title = "Boxplot of {} in terms of {}".format(selected_columns1,selected_columns2)
                            )
                            
                        fig1 = go.Figure(data1,layout1)
                        st.write(fig1)
                        
                        #histogram plot        
                        st.subheader('Histogram plot for {} for all the {}'.format(selected_columns2,selected_columns1))
                        
                        trace0 = go.Histogram(
                                x=df.loc[df[selected_columns1] == typ[0]][selected_columns2], name=typ[0],
                                opacity=0.75
                            )
                        trace1 = go.Histogram(
                                x=df.loc[df[selected_columns1] == typ[1]][selected_columns2], name=typ[1],
                                opacity=0.75
                                
                            )
                       
                        data = [trace0, trace1]
                            
                        layout = go.Layout(barmode='overlay', title="Histogram of {} for all the {}".format(selected_columns2,selected_columns1))
                        fig2 = go.Figure(data=data, layout=layout)
                        st.write(fig2)
            
        if length>= 3:
            
            if st.checkbox("click to see the Results for Selected Columns"):
                
                
                st.markdown("**Introduction**")
                
                st.write("""An analysis of variance (ANOVA) was conducted to determine whether there were significant differences in {} by {}""".format(selected_columns2,selected_columns1))            
                st.markdown("**Results**")
                from pylab import rcParams
                import statsmodels.api as sm
                from statsmodels.formula.api import ols
                import warnings            
                warnings.filterwarnings("ignore")
                rcParams['figure.figsize'] = 20,10
                rcParams['font.size'] = 30
                import numpy as np
                np.random.seed(8)
                import pingouin as pg
                
                aov = pg.anova(dv=selected_columns2, between=selected_columns1, data=df,effsize="n2",detailed=True)
                
                st.write(aov.round(3))
                
                st.write("""- The ANOVA was examined based on an alpha value of .05. The results of the ANOVA were significant, F({}, {}) = {}, p = {}, indicating there were significant differences in {} among the levels of {} . The eta squared was {} indicating Academic_area_expertise explains approximately {}% of the variance in {}. 
- The means and standard deviations are presented in Table below""".format(aov['DF'][0],aov['DF'][1],aov['F'][0].round(3),aov['p-unc'][0].round(3),selected_columns1,selected_columns2,aov['n2'][0].round(3),aov['n2'][0].round(2)*100,selected_columns2))
                st.write("**✌🏼Mean, Standard Deviation, and Sample Size for DP by Academic_area_expertise**")
                stat=df.groupby(selected_columns1)[selected_columns2].describe()
                
                stat['skewness']=df.groupby(selected_columns1)[selected_columns2].skew()
                stat['SEM']=df.groupby(selected_columns1)[selected_columns2].sem()
                stat['N']=df.groupby(selected_columns1)[selected_columns2].value_counts().sum()
                st.write(stat) 
                
                
                st.write("**✌🏼Pairwise Tukey post-hocs on selected columns**")
                phoc=df.pairwise_tukey(dv=selected_columns2, between= selected_columns1).round(3)
                def determine_diff(diff):
                
                    if diff > 0 :
                        return 'larger'
                    elif diff == 0 : 
                        return 'equal'
                    elif diff < 0 :
                        return 'lesser'
                phoc["difference"] = phoc["diff"].apply(lambda x: determine_diff(x))     
                st.write(phoc)
                
                dlen=len(phoc.index)
                
                st.subheader("**Results for Pairwise Tukey post-hocs**")
                st.write("""Paired t-tests were calculated between each pair of measurements 
    to further examine the differences among the variables based on an alpha of .05. 
    The Tukey HSD p-value adjustment was used to correct for the effect of multiple comparisons on the family-wise error rate.
    For the main effect of {}, the mean of {} for different pairs are as follows.""".format(selected_columns1,selected_columns2))
                for i in range(dlen):
                    st.write("""
                             
👉 For {} Mean {} was significantly {}  for {} Mean {} with difference of {} and standard error {} and T-values {} with Tukey-HSD corrected p-values {} with Hedges effect size {}""".format(dlen,selected_columns1,selected_columns2,phoc['A'][i],phoc['mean(A)'][i],phoc['difference'][i],phoc['B'][i],phoc['mean(B)'][i],phoc['diff'][i],phoc['se'][i],phoc['T'][i],phoc['p-tukey'][i],phoc['hedges'][i]))
           
                st.write("**Means of {} by {} with 95.00% CI Error Bars**".format(selected_columns2,selected_columns1))
                import plotly.express as px
                fig = px.bar(stat, x=stat.index, y="mean", color="count")
                st.write(fig)
                
            
            
            
            
            
            
            
        
            
            
        
        
        
        
        
        
        
    st.text('© ASIA Consulting 2022') 
            




if __name__=='__main__':
    main()
