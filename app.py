# Laget av Marius Fossen 
import pandas as pd
import subprocess
from io import StringIO
import uuid
import dash_uploader as du
import dash
from dash import dcc, html, Input, Output, State, Patch
import os, shutil
from dash.exceptions import PreventUpdate
import plotly_express as px
import dash_bootstrap_components as dbc
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template
import plotly.io as pio
import os
import plotly.figure_factory as ff
#import matplotlib.pyplot as plt

load_figure_template('bootstrap')

UPLOAD_FOLDER_ROOT = 'Uploads'
if not os.path.exists(UPLOAD_FOLDER_ROOT):
    os.makedirs(UPLOAD_FOLDER_ROOT)
    os.chmod(UPLOAD_FOLDER_ROOT, 0o777)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
                suppress_callback_exceptions=True)

app.server.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB

du.configure_upload(app, UPLOAD_FOLDER_ROOT)

app.title = 'Monte Carlo Simulering'
server = app.server

def format_with_space(number):
    return '{:,.0f}'.format(number).replace(',', ' ')

def get_upload_component(id):
    return du.Upload(
        id=id,
        text='Slipp filen her eller trykk for å laste opp EFFEKT base',
        text_completed='Lastet opp: ',
        cancel_button=True,
        max_files=1,
        max_file_size=2000,  # 2000 Mb
        filetypes=['MDB', 'mdb', 'accdb'],
        upload_id=uuid.uuid1(),  # Unique session id
    )



def get_app_layout():
    return dbc.Container([
        dbc.Alert(["Dette er en testversjon. Bruk resultatene med omhu. Tilbakemeldinger? Send en ", 
                   html.A('epost.', href='mailto:marius.fossen@vegvesen.no?subject=Tilbakemelding Monte Carlo simulering', className="alert-link"), ], color="warning"),
        dbc.Row([
            dcc.Store(id='memory', storage_type='memory'),
            dbc.Col(html.H1('Monte Carlo Simulering', className='text-center mt-4'))
        ]),
        dbc.Row([
            dcc.Store(id='memory1', storage_type='memory'),
            dbc.Col(dcc.Markdown('''### Introduksjon:
Denne applikasjonen lar deg utføre Monte Carlo-simuleringer på EFFEKT-databaser. Ved å laste opp en database og kjøre simuleringer, kan du evaluere prosjektalternativer basert på en rekke variabler som påvirker kostnader, nytte og risiko. Simuleringen tar hensyn til usikkerheter i faktorer som **trafikantnytte**, **drift og vedlikehold**, **ulykkeskostnader** og **investeringskostnader** for å gi et estimat over nettonåverdi (NNV) og nettonåverdi per budsjettkrone (NNB).

Ved å utføre tusenvis av simuleringer med varierende inngangsdata, gir modellen et bilde av prosjektets usikkerheter og potensielle utfall. Dette gir beslutningstakere bedre innsikt i hvordan ulike faktorer påvirker de økonomiske konsekvensene av prosjektet.

#### Slik bruker du applikasjonen:
1. Last opp en EFFEKT-database.
2. Velg et eller flere prosjektalternativer fra nedtrekksmenyen.
3. Se resultatene for nettonåverdi (NNV) og nettonåverdi per budsjettkrone (NNB) i form av tabeller og grafer.
''', className='text-body mt-4', id='text-body'))
        ]),
        dbc.Row([
            dcc.Store(id='memory2',storage_type ='memory'),
            dbc.Col(dcc.Dropdown(options={}, multi=True, id='dropdown', placeholder='Velg alternativ'), className='mb-auto')
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Collapse(
                    dbc.Accordion([
                        dbc.AccordionItem([
                            dbc.Col(dcc.Loading(id='loading-1', type='default', children=html.Div(id='tabell')), width=12),
                            dbc.Col(dcc.Loading(id='loading-2', type='default', children=html.Div(id='Histo')), width=12),
                        ], title="NNV (Nettonåverdi)"),
                        dbc.AccordionItem([
                            dbc.Col(dcc.Loading(id='loading-3', type='default', children=html.Div(id='tabell_nnb')), width=12),
                            dbc.Col(dcc.Loading(id='loading-4', type='default', children=html.Div(id='Histo2')), width=12),
                        ], title="NNB (Nettonåverdi pr budsjettkrone)"),
                    ], always_open=True),
                    id='accordion-collapse',
                    is_open=False
                )
            )
        ]),
        dbc.Row([
            dbc.Col(dcc.Markdown(id='text-body2', className='text-body mt-4'))
        ]),
        dbc.Row([
            dbc.Col(dcc.Markdown(id='hypothesis-result'))  # Add this row to display the hypothesis test result
        ]),
        dbc.Row([
            dbc.Col(get_upload_component(id='dash-uploader'), className='m-5')
        ]),        
        dbc.Row([
            dbc.Col(
                html.Div([
                    dbc.Button("Les mer om Monte Carlo simulering", id="open", n_clicks=0, className="mb-5"),
                    dbc.Modal([
                        dbc.ModalHeader(dbc.ModalTitle("Monte Carlo simulering")),
                        dbc.ModalBody(dcc.Markdown('''Monte Carlo-simulering er en kraftig numerisk metode som brukes for å løse komplekse problemer ved hjelp av tilfeldige tall. Metoden er oppkalt etter det verdensberømte kasinoet i Monaco, og den involverer bruk av tilfeldig genererte tall for å modellere usikkerhet og utføre gjentatte eksperimenter
Simuleringen utføres ved å gjenta eksperimentet tusenvis eller millioner av ganger, hver gang med tilfeldige variabler som innganger.
                                                   
Metoden er bredt anvendelig og brukes i en rekke fagområder, inkludert finans, ingeniørvitenskap, fysikk, biologi og datavitenskap. 
Eksempler på bruksområder inkluderer evaluering av risiko i investeringer, optimalisering av produksjonsprosesser, og forutsigelse av komplekse systemers atferd.

Monte Carlo-simulering gir muligheten til å håndtere usikkerhet og kompleksitet på en robust måte, og den har blitt en verdifull tilnærming for problemløsning i moderne vitenskap og industri.''')),
                        dbc.ModalFooter(dbc.Button("Lukk", id="close", className="ms-auto", n_clicks=0))
                    ], id="modal", size='xl', is_open=False),
                ]),
            ),
        ]),        
    ])

@app.callback(
    Output('accordion-collapse', 'is_open'),
    [Input('dropdown', 'value')],
    [State('accordion-collapse', 'is_open')]
)
def toggle_accordion(dropdown_value, is_open):
    if dropdown_value and not is_open:
        return True
    return is_open


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

def std_dev_confidence_interval(data):
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    std_dev = np.std(a, ddof=1)
    h = 1.96 * std_dev
    return m, m - h, m + h

@app.callback(
    [Output('Histo', 'children'),
     Output('tabell', 'children'),
     Output('Histo2', 'children'),
     Output('tabell_nnb', 'children'),
     Output('hypothesis-result', 'children'),
     Output("text-body2", 'children')],
    [Input('dropdown', 'value'),
     State('memory', 'data'),
     State('memory1', 'data'),
     State('memory2','data')]
)
def update_graph(dropdown, data, data2, data3):
    if not dropdown:
        return [], [], [], [], [], [] 

    df = pd.DataFrame(data)
    #print('df1 head',df.head())
    Prosjekt = pd.DataFrame(data2)
    
    #print(Prosjekt.head())
    df2 = pd.DataFrame(data3)
    print('df2 head',df2.head())
    Navn = Prosjekt.iat[0, 1]
    Kalkrente = Prosjekt.iat[0, 3]
    Prisnivå = Prosjekt.iat[0, 4]
    Sammenligningsår = Prosjekt.iat[0, 5]
    Levetid = Prosjekt.iat[0, 6]
    Ansvarlig = Prosjekt.iat[0, 45]

    df1 = df[dropdown].describe()
    df1.index = ['Antall simuleringer', 'Gjennomsnitt', 'Standardavvik', 'Minimumsverdi', '25% kvantil', '50% kvantil', '75% kvantil', 'Maksimumsverdi']
    df_nnb = pd.DataFrame(data3)
    df_nnb1 = df_nnb[dropdown].describe()
    df_nnb1.index = ['Antall simuleringer', 'Gjennomsnitt', 'Standardavvik', 'Minimumsverdi', '25% kvantil', '50% kvantil', '75% kvantil', 'Maksimumsverdi']

    # 95% ki
    confidence_intervals_std = {}
    confidence_intervals_std_nnb = {}
    for col in dropdown:
        m_std, lower_std, upper_std = std_dev_confidence_interval(df[col])
        confidence_intervals_std[col] = {'Lower': lower_std, 'Upper': upper_std}
        
        m_std_nnb, lower_std_nnb, upper_std_nnb = std_dev_confidence_interval(df_nnb[col])
        confidence_intervals_std_nnb[col] = {'Lower': lower_std_nnb, 'Upper': upper_std_nnb}


    # KI for data
    confidence_data_std = {col: [confidence_intervals_std[col]['Lower'], confidence_intervals_std[col]['Upper']] for col in dropdown}
    confidence_df_std = pd.DataFrame(confidence_data_std, index=['95% KI nedre', '95% KI øvre'])
    df1 = pd.concat([df1, confidence_df_std])

    confidence_data_std_nnb = {col: [confidence_intervals_std_nnb[col]['Lower'], confidence_intervals_std_nnb[col]['Upper']] for col in dropdown}
    confidence_df_std_nnb = pd.DataFrame(confidence_data_std_nnb, index=['95% KI nedre', '95% KI øvre'])
    df_nnb1 = pd.concat([df_nnb1, confidence_df_std_nnb])
    
    df1 = df1.map(lambda x: format_with_space(x) if isinstance(x, (int, float)) else x)
    df_nnb1.loc['Antall simuleringer'] = df_nnb1.loc['Antall simuleringer'].apply(lambda x: format_with_space(x) if isinstance(x, (int, float)) else x)
    df_nnb1 = df_nnb1.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

   

    tabell = dbc.Table.from_dataframe(df1.reset_index().rename(columns={'index': 'Deskriptiv analyse (NNV)'}).round(2), striped=True, bordered=True, hover=True)
    tabell_nnb = dbc.Table.from_dataframe(df_nnb1.reset_index().rename(columns={'index': 'Deskriptiv analyse (NNB)'}).round(2), striped=True, bordered=True, hover=True)

    
    #tabell = [dbc.Col(dbc.Table.from_dataframe(data4, striped=True, bordered=True, hover=True), className='mb-auto')]
    fig1 = px.ecdf(df[dropdown], marginal='histogram')
    """for i, col in enumerate(dropdown):
        lower = confidence_intervals_std[col]['Lower']
        upper = confidence_intervals_std[col]['Upper']
        fig1.add_shape(type="line", x0=lower, y0=0, x1=lower, y1=1, line=dict(color='red', dash="dash"))
        fig1.add_shape(type="line", x0=upper, y0=0, x1=upper, y1=1, line=dict(color='green', dash="dash"))"""

    
    #fig1 = px.ecdf(df[dropdown], marginal='histogram')

    # Vertikale linjer for KI
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate(dropdown):
        lower = confidence_intervals_std[col]['Lower']
        upper = confidence_intervals_std[col]['Upper']
        color = colors[i % len(colors)]
        fig1.add_shape(
            type="line",
            x0=lower,
            y0=0,
            x1=lower,
            y1=1,
            line=dict(color=color, width=2, dash="dashdot"),
            name=f"95% KI nedre {col}"
        )
        fig1.add_trace(
            go.Scatter(
                x=[lower], y=[0.5],
                mode="markers",
                marker=dict(color=color, size=10, symbol="line-ns-open"),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"95% KI nedre, {col}: {lower:,.0f},- {Prisnivå} Kroner"
            )
        )
        fig1.add_shape(
            type="line",
            x0=upper,
            y0=0,
            x1=upper,
            y1=1,
            line=dict(color=color, width=2, dash="dashdot"),
            name=f"95% KI øvre {col}"
        )
        fig1.add_trace(
            go.Scatter(
                x=[upper], y=[0.5],
                mode="markers",
                marker=dict(color=color, size=10, symbol="line-ns-open"),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"95% KI øvre, {col}: {upper:,.0f},- {Prisnivå} Kroner"
            )
        )

    fig1.update_layout(
        xaxis_title='%s Kroner' % Prisnivå,
        yaxis_title='Sannsynlighet',
        legend_title='',
        height=650,
        legend=dict(yanchor='bottom', y=-0.2, orientation='h', xanchor='left', x=0,),
        hovermode="x unified"
    )
    # Graphs for NNB
    fig2 = px.ecdf(df_nnb[dropdown], marginal='histogram')
    """for i, col in enumerate(dropdown):
        lower_nnb = confidence_intervals_std_nnb[col]['Lower']
        upper_nnb = confidence_intervals_std_nnb[col]['Upper']
        fig2.add_shape(type="line", x0=lower_nnb, y0=0, x1=lower_nnb, y1=1, line=dict(color='red', dash="dash"))
        fig2.add_shape(type="line", x0=upper_nnb, y0=0, x1=upper_nnb, y1=1, line=dict(color='green', dash="dash"))"""
    colors = px.colors.qualitative.Plotly
    for i, col in enumerate(dropdown):
        lower_nnb = confidence_intervals_std_nnb[col]['Lower']
        upper_nnb = confidence_intervals_std_nnb[col]['Upper']
        color = colors[i % len(colors)]
        fig2.add_shape(
            type="line",
            x0=lower_nnb,
            y0=0,
            x1=lower_nnb,
            y1=1,
            line=dict(color=color, width=2, dash="dashdot"),
            name=f"95% KI nedre {col}"
        )
        fig2.add_trace(
            go.Scatter(
                x=[lower_nnb], y=[0.5],
                mode="markers",
                marker=dict(color=color, size=10, symbol="line-ns-open"),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"95% KI nedre, {col}: {lower_nnb:,.0f},- {Prisnivå} Kroner"
            )
        )
        fig2.add_shape(
            type="line",
            x0=upper_nnb,
            y0=0,
            x1=upper_nnb,
            y1=1,
            line=dict(color=color, width=2, dash="dashdot"),
            name=f"95% KI øvre {col}"
        )
        fig2.add_trace(
            go.Scatter(
                x=[upper_nnb], y=[0.5],
                mode="markers",
                marker=dict(color=color, size=10, symbol="line-ns-open"),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"95% KI øvre, {col}: {upper_nnb:,.0f},- {Prisnivå} Kroner"
            )
        )

    fig2.update_layout(
        xaxis_title='%s Kroner' % Prisnivå,
        yaxis_title='Sannsynlighet',
        legend_title='',
        height=650,
        legend=dict(yanchor='bottom', y=-0.2, orientation='h', xanchor='left', x=0,),
        hovermode="x unified"
    )

    Graf = dcc.Graph(id='graf1', figure=fig1)
    Graf_nnb = dcc.Graph(id='graf2', figure=fig2)
    """Graf = [
        dbc.Col(dcc.Graph(id='graf1', figure=fig1), className='mb-4'),
    ]"""
    fupper= upper.round(-1)
    fupper= format_with_space(upper)
    flower = lower.round(-1)
    flower = format_with_space(lower)
    fupper_nnb = upper_nnb.round(2)
    
    flower_nnb = lower_nnb.round(2)
    
    bobkaare = '''#### Resultater
Monte Carlo-simuleringen har evaluert de valgte alternativene basert på inputfaktorer med variabel usikkerhet. Resultatene viser både **nettonåverdi (NNV)** og **nettonåverdi per budsjettkrone (NNB)** med tilhørende konfidensintervaller for 95 % sikkerhet. Dette betyr at prosjektets samfunnsøkonomiske verdi sannsynligvis vil ligge innenfor disse grensene med en sannsynlighet på 95 %.

- **NNV**: Nettonåverdi representerer den samlede samfunnsøkonomiske verdien av prosjektet, etter at alle kostnader og nyttefaktorer er tatt i betraktning.
- **NNB**: Nettonåverdi per budsjettkrone viser den samfunnsøkonomiske avkastningen per investert krone.

**Konklusjon**: Hvis konfidensintervallene indikerer at prosjektet har en positiv NNV, kan dette bety at prosjektet vil gi en samfunnsøkonomisk gevinst. På den andre siden, hvis intervallet inkluderer negative verdier, bør prosjektet vurderes med forsiktighet.
'''

    # Hypotesetest
    hypothesis_result = ""
    if len(dropdown) == 2:  
        col1, col2 = dropdown
        sample1 = df[col1]
        sample2 = df[col2]
        shapiro_test_sample1 = stats.shapiro(sample1)
        shapiro_test_sample2 = stats.shapiro(sample2)

        ks_test_sample1 = stats.kstest(sample1, 'norm', args=(np.mean(sample1), np.std(sample1)))
        ks_test_sample2 = stats.kstest(sample2, 'norm', args=(np.mean(sample2), np.std(sample2)))
        print(f"Shapiro-Wilk Test Sample 1: Statistic={shapiro_test_sample1[0]}, p-value={shapiro_test_sample1[1]}")
        if shapiro_test_sample1[1] > 0.05:
            normsh1='følger en normalfordeling'
            
        else:
            normsh1='følger ikke en normalfordeling'
            

        print(f"Shapiro-Wilk Test Sample 2: Statistic={shapiro_test_sample2[0]}, p-value={shapiro_test_sample2[1]}")
        if shapiro_test_sample2[1] > 0.05:
            normsh2='følger en normalfordeling'
            
        else:
            normsh2='følger ikke en normalfordeling'
            

        print(f"Kolmogorov-Smirnov Test Sample 1: Statistic={ks_test_sample1[0]}, p-value={ks_test_sample1[1]}")
        if ks_test_sample1[1] > 0.05:
            normks1='følger en normalfordeling'
            
        else:
            normks1='følger ikke en normalfordeling'
            

        print(f"Kolmogorov-Smirnov Test Sample 2: Statistic={ks_test_sample2[0]}, p-value={ks_test_sample2[1]}")
        if ks_test_sample2[1] > 0.05:
            normks2='følger en normalfordeling'
            
        else:
            normks2='følger ikke en normalfordeling'
            
        normal_sample1 = (shapiro_test_sample1[1] > 0.05) and (ks_test_sample1[1] > 0.05)
        normal_sample2 = (shapiro_test_sample2[1] > 0.05) and (ks_test_sample2[1] > 0.05)

        #if normal_sample1 and normal_sample2:
            # Sjekk for lik varians
        equal_var = np.var(sample1) == np.var(sample2)

        # t-test
        if equal_var:
            t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=True)
        else:
            t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)

        # Cohen's d
        mean1 = np.mean(sample1)
        mean2 = np.mean(sample2)
        std1 = np.std(sample1, ddof=1)
        std2 = np.std(sample2, ddof=1)

        n1 = len(sample1)
        n2 = len(sample2)

        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohen_d = (mean1 - mean2) / pooled_std
        cohen_d = cohen_d.round(4)

        # KI for gjennomsnitt
        conf_interval_1 = stats.t.interval(0.95, n1 - 1, loc=mean1, scale=std1/np.sqrt(n1))
        conf_interval_2 = stats.t.interval(0.95, n2 - 1, loc=mean2, scale=std2/np.sqrt(n2))

        # KI for mean difference
        mean_diff = mean1 - mean2
        se_diff = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        conf_interval_diff = stats.t.interval(0.95, df=min(n1, n2) - 1, loc=mean_diff, scale=se_diff)

        # print(f"\nt-statistic: {t_stat}")
        # print(f"p-value: {p_value}")
        # print(f"Effect Size (Cohen's d): {cohen_d}")
        # Print conclusion based on Cohen's d
        if abs(cohen_d) < 0.2:
            effect_size_conclusion = "Dette indikerer en veldig liten effektstørrelse."
        elif abs(cohen_d) < 0.5:
            effect_size_conclusion = "Dette indikerer en liten effektstørrelse."
        elif abs(cohen_d) < 0.8:
            effect_size_conclusion = "Dette indikerer en middels effektstørrelse."
        else:
            effect_size_conclusion = "Dette indikerer en stor effektstørrelse."

        print(equal_var)
        hypothesis_result = f'''#### Hypotesetest {'(T test)' if equal_var else '(Welch t test)'} 
            
**Nullhypotese t-test(H0)**: Det er ingen forskjell i snittet mellom {col1} og {col2}.

**Alternativ hypotese t-test(H1)**: Det er en signifikant forskjell i snittet mellom {col1} og {col2}.  

***Konklusjon***: {'Avvis' if p_value < 0.05 else 'Kan ***ikke*** avvise'} nullhypotesen for 0.05 signifikansnivå.
        '''
# **t-statistic**: {t_stat:.4e}  
# **p-value**: {p_value:.2e} 
        """else:
            # Mann-Whitney U test
            n1 = len(sample1)
            n2 = len(sample2)
            r = 1 - (2 * u_stat) / (n1 * n2)
            if abs(r) < 0.1:
                konkr = "Dette indikerer en veldig liten effektstørrelse."
            elif abs(r) < 0.3:
                konkr = "Dette indikerer en liten effektstørrelse."
            elif abs(r) < 0.5:
                konkr = "Dette indikerer en middels effektstørrelse."
            else:
                konkr = "Dette indikerer en stor effektstørrelse."
            u_stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
            hypothesis_result = f'''#### Hypotesetest (Mann-Whitney U test)
Det ble gjort en Mann Whitney U test fordi et eller begge alternativer ser ikke ut til å være normalfordelte:

***Shapiro test:*** {col1} {normsh1}  
***Shapiro test:*** {col2} {normsh2}  
***Kolmogorov-Smirnov Test:*** {col1} {normks1}  
***Kolmogorov-Smirnov Test:*** {col2} {normks2}  

**Nullhypotese t-test(H0)**: Det er ingen forskjell i rangsummen mellom {col1} og {col2}.

**Alternativ hypotese t-test(H1)**: Det er en signifikant forskjell i rangsummen mellom {col1} og {col2}.  

**t-statistic**: {u_stat:.4e}  
**p-value**: {p_value:.2e}  

***Konklusjon***: {'Avvis' if p_value < 0.05 else 'Kan ***ikke*** avvise'} nullhypotesen for 0.05 signifikansnivå.
        '''

            print("\nMann-Whitney U Test")
            print(f"U-statistic: {u_stat}")
            print(f"p-value: {p_value}")"""

        sa1=str(col1)
        sa2=str(col2)
        fig2 = ff.create_distplot([sample2,sample1], group_labels=[sa1, sa2], show_hist=False, show_rug=False)
        fig2.update_layout(
        xaxis_title='Verdi' % Prisnivå,
        yaxis_title='Tetthet',
        legend_title='',
        height=650,
        legend=dict(yanchor='bottom', y=-0.2, orientation='h', xanchor='left', x=0,),
        hovermode="x unified")
   
        Graf2 =[
            dbc.Col(dcc.Graph(id='graf2', figure=fig2), className='mb-4'),
        ]
        
        

        
                        
    else:
        hypothesis_result = "#### Hypotesetest\nVelg to alternativer for å gjennomføre en hypotesetest."
        Graf2 = []
    return Graf, tabell, Graf_nnb, tabell_nnb, hypothesis_result, bobkaare

app.layout = get_app_layout

@du.callback(
    output=(Output("dropdown", "options"),
            Output("memory", "data"),
            Output("memory1", "data"),
            Output("text-body", "children"),
            Output("memory2", "data")),
)
def callback_on_completion(status: du.UploadStatus):
    pd.set_option('future.no_silent_downcasting', True)    
    simulations = []
    simb = []
    prosjekter = []
    Sti = status.uploaded_files[0]
    print(Sti)
    def export_table_to_dataframe(database_path, table_name):
        command = ['mdb-export', database_path, table_name]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        csv_content = stdout.decode('utf-8')
        return pd.read_csv(StringIO(csv_content))
    query = export_table_to_dataframe('{}'.format(Sti), 'TotKostPlanlagt')
    sql2 = export_table_to_dataframe('{}'.format(Sti), 'TotKostAlt0')
    # print(sql2, '{}'.format(Sti))
    sql3 = export_table_to_dataframe('{}'.format(Sti), 'Prosjekt')
    sql4 = export_table_to_dataframe('{}'.format(Sti), 'Planperioder')
    sql5 = export_table_to_dataframe('{}'.format(Sti), 'Vegnett')
    sql6 = export_table_to_dataframe('{}'.format(Sti), 'Utbyggingsplan')

    Tiltak = query
    Referanse = sql2
    ProsjektPlan = sql3
    Planperioder = sql4
    Vegnett = sql5
    Utbyggingsplan = sql6

    UnikeProsjekt = ProsjektPlan.Nr.unique().tolist() # Kan inneholde flere veinett, må inneholde prosjekt Sjekk utbyggingsplan
    Navn = ProsjektPlan.iat[0, 1]
    Kalkrente = ProsjektPlan.iat[0, 3]
    Prisnivå = ProsjektPlan.iat[0, 4]
    Sammenligningsår = ProsjektPlan.iat[0, 5]
    Levetid = ProsjektPlan.iat[0, 6]
    Ansvarlig = ProsjektPlan.iat[0, 45]
    Intervall = []

    def format_with_space(number):
        return '{:,.0f}'.format(number).replace(',', ' ')

    for j in UnikeProsjekt:
        pro = ProsjektPlan.loc[ProsjektPlan.Nr == j]
        prosjetnavn = pro.iat[0, 1]
        prosjekter.append(pro)
        Utbyggingsplan1 = Utbyggingsplan.loc[Utbyggingsplan.ProNr == j]
        UnikeUtbyggingsplaner = Utbyggingsplan1.Nr.unique().tolist()
        dfref = Referanse.query('ProNr == {} & Alternativ == 0'.format(j))
        Referansekostnader = dfref.iloc[:, 4:23].fillna(0)
        EFFEKTRef = Referansekostnader.sum()
        TotalReferansekostnader = Referansekostnader.sum()
        DogVRef = Referansekostnader['Drift_vedlikehold'].sum()
        UlykkerRef = Referansekostnader['Ulykker'].sum()
        ReferanseNytte = (Referansekostnader['Kjøretøykostnader'].sum() +
                          Referansekostnader['Direkteutgifter'].sum() +
                          Referansekostnader['Tidskostnader'].sum() +
                          Referansekostnader['Nyskapt'].sum() +
                          Referansekostnader['Ulempeskostnader'].sum() +
                          Referansekostnader['Helsevirkninger'].sum() +
                          Referansekostnader['Utrygghetskostnader'].sum() +
                          Referansekostnader['Operatørkostnader'].sum() +
                          Referansekostnader['Operatørinntekter'].sum() +
                          Referansekostnader['Operatøroverføringer'].sum() +
                          Referansekostnader['Offentlige_overføringer'].sum() +
                          Referansekostnader['Skatte_avgiftsinntekter'].sum() +
                          Referansekostnader['Støy_luft'].sum() +
                          Referansekostnader['Andre_kostnader'].sum() +
                          Referansekostnader['Skattekostnad'].sum())
        for i in UnikeUtbyggingsplaner:
            plannavn = Utbyggingsplan1.loc[Utbyggingsplan1.Nr == i]
            plannavn = plannavn.iat[0, 2]
            dftil = Tiltak.query('ProNr == {} & PlanNr == {} & Alternativ == 0'.format(j, i))
            Tiltakskostnader = dftil.iloc[:, 4:24].fillna(0)
            TotalTiltakskostnader = Tiltakskostnader.sum()
            EFFEKTTil = Tiltakskostnader.sum()
            TrafnytteTil = Tiltakskostnader['Trafikantnytte'].sum()
            DogVTil = Tiltakskostnader['Drift_vedlikehold'].sum()
            UlykkerTil = Tiltakskostnader['Ulykker'].sum()
            Investring = Tiltakskostnader['Investeringer'].sum()
            TiltakNytte = (Tiltakskostnader['Kjøretøykostnader'].sum() +
                           Tiltakskostnader['Direkteutgifter'].sum() +
                           Tiltakskostnader['Tidskostnader'].sum() +
                           Tiltakskostnader['Nyskapt'].sum() +
                           Tiltakskostnader['Ulempeskostnader'].sum() +
                           Tiltakskostnader['Helsevirkninger'].sum() +
                           Tiltakskostnader['Utrygghetskostnader'].sum() +
                           Tiltakskostnader['Operatørkostnader'].sum() +
                           Tiltakskostnader['Operatørinntekter'].sum() +
                           Tiltakskostnader['Operatøroverføringer'].sum() +
                           Tiltakskostnader['Offentlige_overføringer'].sum() +
                           Tiltakskostnader['Skatte_avgiftsinntekter'].sum() +
                           Tiltakskostnader['Støy_luft'].sum() +
                           Tiltakskostnader['Andre_kostnader'].sum() +
                           Tiltakskostnader['Restverdi'].sum() +
                           Tiltakskostnader['Skattekostnad'].sum() +
                           Tiltakskostnader['Restverdi'].sum())
            kostnader_tiltak = Tiltakskostnader['Investeringer'].sum() + Tiltakskostnader['Drift_vedlikehold'].sum() + Tiltakskostnader['Offentlige_overføringer'].sum() + Tiltakskostnader['Skatte_avgiftsinntekter'].sum()
            kostnader_referanse = Referansekostnader['Drift_vedlikehold'].sum() + Referansekostnader['Offentlige_overføringer'].sum() + Referansekostnader['Skatte_avgiftsinntekter'].sum()
            diff_kostnader = abs(kostnader_tiltak - kostnader_referanse)
            diff = TiltakNytte - ReferanseNytte
            Effekt = EFFEKTTil - EFFEKTRef
            correlation_matrix = np.array([[1, 0.05, 0.1, 0], [0.05, 1, 0, 0], [0.1, 0, 1, 0], [0, 0, 0, 1]])
            testing4 = pd.DataFrame(correlation_matrix)
            std_dev = np.array([1, 1, 1, 1])           

            num_samples = 50000

            def show_func(d):
                funksjoner = {
                    'normalTR': np.random.default_rng().normal(1, 0.220388, size=(num_samples, 1)),
                    'normal1UL': np.random.default_rng().normal(1, 0.075095, size=(num_samples, 1)),
                    'normal2DV': np.random.default_rng().normal(1, 0.18, size=(num_samples, 1)),
                    'triangular': np.random.default_rng().triangular(0.89, 1, 1.4, size=(num_samples, 1)),
                    'lognormal': np.random.default_rng().lognormal(1, 0.1, size=(num_samples, 1))
                }
                return funksjoner[d]

            def covariance_from_correlation(correlation_matrix):
                eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
                sqrt_eigenvalues = np.sqrt(np.diag(eigenvalues))
                inv_sqrt_eigenvalues = np.linalg.inv(sqrt_eigenvalues)
                covariance_matrix1 = np.dot(np.dot(eigenvectors, sqrt_eigenvalues), eigenvectors.T)
                return covariance_matrix1

            def correlation_to_covariance(correlation_matrix, std_dev):
                covariance_matrix = np.outer(std_dev, std_dev) * correlation_matrix
                return covariance_matrix

            def cholesky_decomposition_with_correlation(correlation_matrix, std_dev):
                covariance_matrix = correlation_to_covariance(correlation_matrix, std_dev)
                # print(covariance_matrix)
                cov1 = covariance_from_correlation(correlation_matrix)
                # print(cov1)
                L = np.linalg.cholesky(cov1)
                return L

            def generate_correlated_samples(num_samples, cholesky_matrix):
                Tr = show_func('normalTR')
                DV = show_func('normal2DV')
                Ul = show_func('normal1UL')
                In = show_func('triangular')
                ukorr = np.concatenate((Tr, DV, Ul, In), axis=1)
                ukorr1 = np.column_stack((ukorr))
                testing = pd.DataFrame(ukorr)
                correlated_samples = np.dot(ukorr, cholesky_matrix.T)
                testing2 = pd.DataFrame(correlated_samples)
                return correlated_samples

            def example_function(samples):
                df = pd.DataFrame(samples)
                df['nnv'] = (TrafnytteTil * df[0] + (DogVTil - DogVRef) * df[1] + (UlykkerTil - UlykkerRef) * df[2] + Investring * df[3] + diff)
                df['nnb'] = (TrafnytteTil * df[0] + (DogVTil - DogVRef) * df[1] + (UlykkerTil - UlykkerRef) * df[2] + Investring * df[3] + diff) / diff_kostnader
                df.columns = ['Trafikantnytte', 'Drift og vedlikehold', 'Ulykker', 'Investering', 'NNV', 'NNB']
                return df[['NNV', 'NNB']] 
            
            cholesky_matrix = cholesky_decomposition_with_correlation(correlation_matrix, std_dev)
            correlated_samples = generate_correlated_samples(num_samples, cholesky_matrix)
            dfnn = example_function(correlated_samples)            
            dfnn = pd.DataFrame(dfnn)
            dfnnv = dfnn['NNV']
            dfnnv = pd.DataFrame(dfnnv)
            dfnnb = dfnn['NNB']
            dfnnb = pd.DataFrame(dfnnb)
            dfnnv = dfnnv.rename(columns={"NNV": "Prosjekt: {}, Utbyggingsplan: {} ".format(prosjetnavn, plannavn)})
            # print(dfnnv)
            dfnnb = dfnnb.rename(columns={"NNB": "Prosjekt: {}, Utbyggingsplan: {} ".format(prosjetnavn, plannavn)})
            #print(dfnnb)
            simulations.append(dfnnv)
            
            simb.append(dfnnb)
            KI = std_dev_confidence_interval(dfnnv)
            KIb = std_dev_confidence_interval(dfnnb)
            # print(KI)

    simulations = pd.concat(simulations, axis=1)
    #print(simulations)
    simulationsb = pd.concat(simb, axis=1)
    # print(simulationsb)
    antall = format_with_space(num_samples)
    li = simulations.columns
    lis = li
    d1 = dict(zip(li, lis))
    simulations = simulations.to_dict()
    simulationsb = simulationsb.to_dict()
    print('nnv data',simulations.keys())
    print('nnb data:', simulationsb.keys())
    Prosjekt = ProsjektPlan.to_dict()
    Bjarne = html.H4('{}'.format(Navn))
    Kjell = '''Prisnivået er {} og det er benyttet en kalkulasjonsrente på {} prosent. Ansvarlig for EFFEKTbasen er {}. Det er gjort {} simuleringer.

Nedenfor må det velges alternativ fra EFFEKTbasen i nedtrekksmenyen. Det er mulig å velge flere alternativer samtidig.

Formålet med Monte Carlo simuleringen er å vise usikkerhetens konsekvens for nettonåverdien til prosjektet. Ved å vise nettonåverdien som et konfidensintervall vil beslutningstager få et mer helhetlig bilde av prosjektetet. 

Parametrene som er variert i simuleringen er:
- **Trafikantnytte**: Normalfordelt med et standardavvik på 22%, 
- **Ulykkeskostnader**: Normalfordelt med et standardavvik på 8%, 
- **Drift & vedlikehold**: Normalfordelt med et standardavvik på 18%, har også en korrelasjonskoffesient med trafikantnytten på 0,57 og ulykker på 0,28
- **Investeringskostnader**: Trekantfordeling som går 11% ned og 20% over.

#### Resultater'''.format(Prisnivå, Kalkrente, Ansvarlig, antall)
    kaare = '''**Akkumulert Sannsynlighet**: Linjeplot som viser den akkumulerte sannsynligheten eller summen over gjentatte simuleringer.'''

#     bobkaare = '''#### Tolkning av resultater
# Resultatene viser at det finnes en viss grad av usikkerhet i både NNV og NNB, med konfidensintervaller som strekker seg fra {} til {} for NNV, og fra {} til {} for NNB.
    
# - **Tolkning av resultater**: Diskuter implikasjonene av simulerte resultater og deres relevans for problemet som blir studert.
# - **Begrensninger og forbedringer**: Identifiser eventuelle begrensninger i simuleringen og mulige forbedringer for fremtidig arbeid.'''

    # Fjerne all data fra mappe
    for filename in os.listdir(UPLOAD_FOLDER_ROOT):
        file_path = os.path.join(UPLOAD_FOLDER_ROOT, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    return d1, simulations, Prosjekt, Kjell, simulationsb

if __name__ == '__main__':
    app.run_server(debug=True)
