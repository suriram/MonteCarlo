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
        filetypes=['MDB', 'mdb'],
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
            dbc.Col(dcc.Markdown('''Her kan du gjøre Monte Carlo simulering av EFFEKTdatabaser. __Last opp EFFEKTdatabasen nedenfor.__ 
Simuleringen varierer trafikantnytten, drift og vedlikehold, investeringskostnader og ulykkeskostnader.''', className='text-body mt-4', id='text-body'))
        ]),
        dbc.Row([
            dbc.Col(dcc.Dropdown(options={}, multi=True, id='dropdown', placeholder='Velg alternativ'), className='mb-auto')
        ]),
        dbc.Row([
            dbc.Col(dcc.Loading(id='loading-1', type='default', children=html.Div(id='tabell')), width=12)
        ]),
        dbc.Row([
            dbc.Col(dcc.Loading(id='loading-2', type='default', children=html.Div(id='Histo')), width=12)
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
Metoden er bredt anvendelig og brukes i en rekke fagområder, inkludert finans, ingeniørvitenskap, fysikk, biologi og datavitenskap. Eksempler på bruksområder inkluderer evaluering av risiko i investeringer, optimalisering av produksjonsprosesser, og forutsigelse av komplekse systemers atferd.
Monte Carlo-simulering gir muligheten til å håndtere usikkerhet og kompleksitet på en robust måte, og den har blitt en verdifull tilnærming for problemløsning i moderne vitenskap og industri.''')),
                        dbc.ModalFooter(dbc.Button("Lukk", id="close", className="ms-auto", n_clicks=0))
                    ], id="modal", size='xl', is_open=False),
                ]),
            ),
        ]),        
    ])

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
     Output('hypothesis-result', 'children')],
    [Input('dropdown', 'value'),
     State('memory', 'data'),
     State('memory1', 'data')]
)
def update_graph(dropdown, data, data1):
    if not dropdown:
        return [], [], []

    df = pd.DataFrame(data)
    Prosjekt = pd.DataFrame(data1)
    Navn = Prosjekt.iat[0, 1]
    Kalkrente = Prosjekt.iat[0, 3]
    Prisnivå = Prosjekt.iat[0, 4]
    Sammenligningsår = Prosjekt.iat[0, 5]
    Levetid = Prosjekt.iat[0, 6]
    Ansvarlig = Prosjekt.iat[0, 45]

    df1 = df[dropdown].describe()
    df1.index = ['Antall simuleringer', 'Gjennomsnitt', 'Standardavvik', 'Minimumsverdi', '25% kvantil', '50% kvantil', '75% kvantil', 'Maksimumsverdi']

    # Calculate 95% confidence intervals for each selected column
    confidence_intervals_std = {}
    for col in dropdown:
        m_std, lower_std, upper_std = std_dev_confidence_interval(df[col])
        confidence_intervals_std[col] = {'Lower': lower_std, 'Upper': upper_std}
        # print(f"{col} - Mean: {m_std}, Std Dev: {np.std(df[col], ddof=1)}, 95% CI 1.96*std: [{lower_std}, {upper_std}]")

    # Add confidence intervals to the dataframe
    confidence_data_std = {col: [confidence_intervals_std[col]['Lower'], confidence_intervals_std[col]['Upper']] for col in dropdown}
    confidence_df_std = pd.DataFrame(confidence_data_std, index=['95% KI nedre', '95% KI øvre'])
    df1 = pd.concat([df1, confidence_df_std])

    data4 = df1.reset_index().rename(columns={'index': 'Deskriptiv analyse (i tusen kroner)'}).round(2)
    data4 = data4.applymap(lambda x: "{:,.0f}".format(x).replace(",", " ") if isinstance(x, (int, float)) else x)
    
    tabell = [dbc.Col(dbc.Table.from_dataframe(data4, striped=True, bordered=True, hover=True), className='mb-auto')]

    fig1 = px.ecdf(df[dropdown], marginal='histogram')

    # Add vertical lines for confidence intervals
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
    Graf = [
        dbc.Col(dcc.Graph(id='graf1', figure=fig1), className='mb-4'),
    ]

    konklusjon = '''### Konklusjon
    
Oppsummering av de viktigste funnene fra Monte Carlo-simuleringen og deres betydning i forhold til problemet som ble studert.'''

    # Hypothesis Test
    hypothesis_result = ""
    if len(dropdown) == 2:  # Perform test only if two columns are selected
        col1, col2 = dropdown

        # Check for NaNs and infinities and remove them
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col1, col2])

        # Ensure columns are single-level and extract data
        if isinstance(df[col1], pd.DataFrame):
            col1_data = df[col1].values.flatten()
        else:
            col1_data = df[col1]
        
        if isinstance(df[col2], pd.DataFrame):
            col2_data = df[col2].values.flatten()
        else:
            col2_data = df[col2]

        # Debugging output
        print(f"Shape of {col1}: {col1_data.shape}, Shape of {col2}: {col2_data.shape}")
        print(f"Head of {col1}:\n{col1_data[:5]}")
        print(f"Head of {col2}:\n{col2_data[:5]}")

        # Ensure data in columns are variable
        if np.unique(col1_data).size == 1 or np.unique(col2_data).size == 1:
            hypothesis_result = f"The data in either {col1} or {col2} is not variable, hypothesis test cannot be performed."
        else:
            # Print the means and standard deviations of the columns for debugging
            print(f"Snittet til {col1}: {col1_data.mean()}, Snittet til {col2}: {col2_data.mean()}")
            print(f"Standardavviket til {col1}: {col1_data.std()}, Standardavviket til {col2}: {col2_data.std()}")
            
            # Check the distributions
            print(f"Fordelingen til {col1}:\n{pd.Series(col1_data).value_counts().head()}")
            print(f"Fordelingen til {col2}:\n{pd.Series(col2_data).value_counts().head()}")

            # Normalize the data if necessary
            col1_data = (col1_data - col1_data.mean()) / col1_data.std()
            col2_data = (col2_data - col2_data.mean()) / col2_data.std()

            # Perform the t-test
            t_stat, p_value = stats.ttest_ind(col1_data, col2_data)
            
            # Debugging output
            print(f"t-statistikk: {t_stat}, p-verdi: {p_value}")
            
            hypothesis_result = f'''#### Hypotesetest(t-test)
**Nullhypotese t-test(H0)**: Det er ingen forskjell i snittet mellom {col1} og {col2}.

**Nullhypotese Mann Whitney U test(H0)**: Det er ingen forskjell i rangsummen mellom {col1} og {col2}.  

**Alternativ hypotese t-test(H1)**: Det er en signifikant forskjell i snittet mellom {col1} og {col2}.  

**Alternativ hypotese Mann Whitney U test(H1)**: Det er en signifikant forskjell i rangsummen mellom {col1} og {col2}. 


**t-statistic**: {t_stat:.4e}  
**p-value**: {p_value:.2e}  

***Konklusjon***: {'Avvis' if p_value < 0.05 else 'Kan ***ikke*** avvise'} nullhypotesen for 0.05 signifikansnivå.
            '''
            # Alternatively, use the Mann-Whitney U test for non-normal data
            u_stat, u_p_value = stats.mannwhitneyu(col1_data, col2_data)
            print(f"Mann Whitney U test: {u_stat}, p-value: {u_p_value}")

            hypothesis_result += f'\n\n**Mann Whitney U test**: {u_stat:.4f}, p-value: {u_p_value:.4e}\n\n'
            hypothesis_result += f"***Konklusjon***: {'Avvis' if u_p_value < 0.05 else 'Kan ***ikke*** avvise'} nullhypotesen for 0.05 signifikansnivå."
            
    else:
        hypothesis_result = "#### Hypotesetest\nVelg to alternativer for å gjennomføre en hypotesetest."

    return Graf, tabell, hypothesis_result

app.layout = get_app_layout

@du.callback(
    output=(Output("dropdown", "options"),
            Output("memory", "data"),
            Output("memory1", "data"),
            Output("text-body", "children"),
            Output("text-body2", 'children')),
)
def callback_on_completion(status: du.UploadStatus):
    pd.set_option('future.no_silent_downcasting', True)    
    simulations = []
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
            diff_kostnader = kostnader_tiltak - kostnader_referanse
            diff = TiltakNytte - ReferanseNytte
            correlation_matrix = np.array([[1, 0.05, 0.1, 0], [0.05, 1, 0, 0], [0.1, 0, 1, 0], [0, 0, 0, 1]])
            testing4 = pd.DataFrame(correlation_matrix)
            std_dev = np.array([1, 1, 1, 1])
            num_samples = 50000

            def show_func(d):
                funksjoner = {
                    'normal': np.random.default_rng().normal(1, 0.3, size=(num_samples, 1)),
                    'uniform': np.random.default_rng().uniform(0.8, 1.2, size=(num_samples, 1)),
                    'triangular': np.random.default_rng().triangular(0.8, 1, 1.4, size=(num_samples, 1)),
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
                Tr = show_func('normal')
                DV = show_func('normal')
                Ul = show_func('normal')
                In = show_func('triangular')
                ukorr = np.concatenate((Tr, DV, Ul, In), axis=1)
                ukorr1 = np.column_stack((ukorr))
                testing = pd.DataFrame(ukorr)
                correlated_samples = np.dot(ukorr, cholesky_matrix.T)
                testing2 = pd.DataFrame(correlated_samples)
                return correlated_samples

            def example_function(samples):
                df = pd.DataFrame(samples)
                df['nnv'] = (TrafnytteTil * df[0] + (DogVTil - DogVRef) * df[1] + (UlykkerTil - UlykkerRef) * df[2] + Investring * df[3] + diff) / diff_kostnader
                df.columns = ['Trafikantnytte', 'Drift og vedlikehold', 'Ulykker', 'Investering', 'NNV']
                return df['NNV']

            cholesky_matrix = cholesky_decomposition_with_correlation(correlation_matrix, std_dev)
            correlated_samples = generate_correlated_samples(num_samples, cholesky_matrix)
            dfnnv = example_function(correlated_samples)
            dfnnv = pd.DataFrame(dfnnv)
            dfnnv = dfnnv.rename(columns={"NNV": "Prosjekt: {}, Utbyggingsplan: {} ".format(prosjetnavn, plannavn)})
            simulations.append(dfnnv)
            KI = std_dev_confidence_interval(dfnnv)
            # print(KI)

    simulations = pd.concat(simulations, axis=1)
    antall = format_with_space(num_samples)
    li = simulations.columns
    lis = li
    d1 = dict(zip(li, lis))
    simulations = simulations.to_dict()
    Prosjekt = ProsjektPlan.to_dict()
    Bjarne = html.H4('{}'.format(Navn))
    Kjell = '''Prisnivået er {} og det er benyttet en kalkulasjonsrente på {} prosent. Ansvarlig for EFFEKTbasen er {}. Det er gjort {} simuleringer.

Nedenfor må det velges alternativ fra EFFEKTbasen i nedtrekksmenyen. Det er mulig å velge flere alternativer samtidig.

Formålet med Monte Carlo simuleringen er å vise usikkerhetens konsekvens for nettonåverdien til prosjektet. Ved å vise nettonåverdien som et konfidensintervall vil beslutningstager få et mer helhetlig bilde av prosjektetet. 

Parametrene som er variert i simuleringen er:
- **Trafikantnytte**: Normalfordelt med et standardavvik på 20%, 
- **Ulykkeskostnader**: Normalfordelt med et standardavvik på 20%, 
- **Drift & vedlikehold**: Normalfordelt med et standardavvik på 20%, har også en korrelasjonskoffesient med trafikantnytten på 0,57 og ulykker på 0,28
- **Investeringskostnader**: Trekantfordeling som går 20% ned og 40% over.

#### Resultater'''.format(Prisnivå, Kalkrente, Ansvarlig, antall)
    kaare = '''**Akkumulert Sannsynlighet**: Linjeplot som viser den akkumulerte sannsynligheten eller summen over gjentatte simuleringer.'''
    bobkaare = '''#### Diskusjon
    
- **Tolkning av resultater**: Diskuter implikasjonene av simulerte resultater og deres relevans for problemet som blir studert.
- **Begrensninger og forbedringer**: Identifiser eventuelle begrensninger i simuleringen og mulige forbedringer for fremtidig arbeid.'''

    # Remove all files in the upload directory
    for filename in os.listdir(UPLOAD_FOLDER_ROOT):
        file_path = os.path.join(UPLOAD_FOLDER_ROOT, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    return d1, simulations, Prosjekt, Kjell, bobkaare

if __name__ == '__main__':
    app.run_server(debug=True)
