import subprocess
import pandas as pd
from io import StringIO

def export_table_to_dataframe(database_path, table_name):
        command = ['mdb-export', database_path, table_name]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        csv_content = stdout.decode('utf-8')
        return pd.read_csv(StringIO(csv_content))

query = export_table_to_dataframe('/home/marius/render/.venv/uploads/eksempel.mdb', 'TotKostPlanlagt')
print(query.columns)