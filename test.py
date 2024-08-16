import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    # Markdown text without the tooltip target
    dcc.Markdown('''
    # Sample Markdown

    This is an example of using a **Tooltip** with Markdown. Hover over the word '''),
    
    # Tooltip target separated from Markdown
    html.Span("tooltip", id="tooltip-target", style={"color": "blue", "font-weight": "bold"}),
    
    # Rest of the markdown text (optional)
    dcc.Markdown(' to see it.'),
    
    # Tooltip definition
    dbc.Tooltip(
        "This is the tooltip text!",
        target="tooltip-target",  # Matches the ID of the html.Span
        placement="top"
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)
