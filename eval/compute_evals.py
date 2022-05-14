# compute_evals.py

# IMPORTS
import json
import plotly.express as px
import plotly.graph_objects as go
# END IMPORTS

# CONSTANTS
MACRO_FILE = "cnc_evals.json"
PANORAMIC_FILE = "pmc_evals.json"
PARTITION_FILE = "pc_evals.json"
CLUSTER_FILE = "kcp_evals.json"

def plot_acc(macro, pan, part, cluster):
    fig = go.Figure()

    epochs = [i for i in range(len(macro['acc']))]
    # Add traces
    fig.add_trace(go.Scatter(x=epochs, y=macro['acc'],
                    mode='markers',
                    name='Conventional-Macro'))
    fig.add_trace(go.Scatter(x=epochs, y=pan['acc'],
                    mode='markers',
                    name='Panoramic-Macro'))
    fig.add_trace(go.Scatter(x=epochs, y=part['acc'],
                    mode='markers',
                    name='Partioned-Local'))
    fig.add_trace(go.Scatter(x=epochs, y=cluster['acc'],
                    mode='markers',
                    name='Cluster-Local'))

    fig.update_layout(
        title="Accuracy Over 10 Epochs",
        xaxis_title="Iterations",
        yaxis_title="Accuracy",
        legend=dict(
        yanchor="bottom",
        y=0,
        xanchor="right",
        x=1
        ))
                    
    fig.write_image('eval_accs.png')

def plot_F1(macro, pan, part, cluster):
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=[i for i in range(len(macro['f1']))], y=macro['f1'],
                    mode='markers',
                    name='Conventional-Macro'))
    fig.add_trace(go.Scatter(x=[i for i in range(len(pan['f1']))], y=pan['f1'],
                    mode='markers',
                    name='Panoramic-Macro'))
    fig.add_trace(go.Scatter(x=[i for i in range(len(part['f1']))], y=part['f1'],
                    mode='markers',
                    name='Partioned-Local'))
    fig.add_trace(go.Scatter(x=[i for i in range(len(cluster['f1']))], y=cluster['f1'],
                    mode='markers',
                    name='Cluster-Local'))

    fig.update_layout(
        title="F1 Evaluation Over 10 Epochs",
        xaxis_title="Iterations",
        yaxis_title="F1 Score",
        legend=dict(
        yanchor="bottom",
        y=0,
        xanchor="right",
        x=1
        ))
                    
    fig.write_image('eval_f1.png')

def plot_loss(macro, pan, part, cluster):
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=[i for i in range(len(macro['loss']))], y=macro['loss'],
                    mode='markers',
                    name='Conventional-Macro'))
    fig.add_trace(go.Scatter(x=[i for i in range(len(pan['loss']))], y=pan['loss'],
                    mode='markers',
                    name='Panoramic-Macro'))
    fig.add_trace(go.Scatter(x=[i for i in range(len(part['loss']))], y=part['loss'],
                    mode='markers',
                    name='Partioned-Local'))
    fig.add_trace(go.Scatter(x=[i for i in range(len(cluster['loss']))], y=cluster['loss'],
                    mode='markers',
                    name='Cluster-Local'))

    fig.update_layout(
        title="Loss Over 10 Epochs",
        xaxis_title="Iterations",
        yaxis_title="Loss (Cross-Entropy)",
        legend=dict(
        yanchor="bottom",
        y=0,
        xanchor="right",
        x=1
        ))
                    
    fig.write_image('eval_loss.png')

# END CONSTANTS
def main():
    macro_f = open(MACRO_FILE)
    macro_data = json.load(macro_f)

    pan_f = open(PANORAMIC_FILE)
    pan_data = json.load(pan_f)

    part_f = open(PARTITION_FILE)
    part_data = json.load(part_f)

    cluster_f = open(CLUSTER_FILE)
    cluster_data = json.load(cluster_f)

    plot_acc(macro_data, pan_data, part_data, cluster_data)
    plot_F1(macro_data, pan_data, part_data, cluster_data)
    plot_loss(macro_data, pan_data, part_data, cluster_data)
if __name__ == "__main__":
    main()