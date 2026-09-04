import Bio.PDB
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pcmap import contactMap
import pandas as pd


pdb_code = "2K7W"
pdb_filename = "data/pdb2k7w.ent"

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    if "CA" not in residue_one or "CA" not in residue_two:
        return np.nan
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float64)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

if __name__ == "__main__":
    # with biopython
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
    model = structure[0]
    dist_matrix = calc_dist_matrix(model["A"], model["B"])
    contact_map = dist_matrix < 12.0
    fig = go.Figure(go.Heatmap(z=contact_map.astype(np.float64),
                            x=[f"{str(res.resname)}:{str(res.id[1])}" for res in model["B"]],
                            y=[f"{str(res.resname)}:{str(res.id[1])}" for res in model["A"]],
                            colorscale='gray',
                            reversescale=True))
    fig.update_layout(
        width=1000,
        height=1000)
    fig.write_html('data/heatmap.html')

    fig = go.Figure(go.Heatmap(z=dist_matrix.astype(np.float64),
                            x=[f"{str(res.resname)}:{str(res.id[1])}" for res in model["B"]],
                            y=[f"{str(res.resname)}:{str(res.id[1])}" for res in model["A"]],
                            colorscale='blues',
                            reversescale=True))
    fig.update_layout(
        width=1000,
        height=1000)
    fig.write_html('data/heatmap_dist.html')



    # with cmap
    c = contactMap(pdb_filename, dist=8)['data']
    root_ids = {f"{item['root']['chainID']}:{item['root']['resID']}" for item in c}
    partner_ids = {f"{item['partners'][i]['chainID']}:{item['partners'][i]['resID']}" for item in c for i in range(len(item['partners']))}
    ids = list(root_ids.union(partner_ids))
    # sort ids: first all IDs that start with A, then all IDs that start with B.
    # after A/B comes a : and then a number. Sort by the number.
    ids = sorted(ids, key=lambda x: (x[0], int(x.split(':')[1])))
    cmap = pd.DataFrame(np.zeros((len(ids), len(ids)), np.float64))
    cmap.index = ids
    cmap.columns = ids
    for i in range(len(c)):
        row_id = c[i]['root']['chainID'] + ":" + c[i]['root']['resID']
        for j in range(len(c[i]['partners'])):
            row_id2 = c[i]['partners'][j]['chainID'] + ":" + c[i]['partners'][j]['resID']
            cmap.at[row_id, row_id2] = 1
            cmap.at[row_id2, row_id] = 1

    # filter such that index only starts with 'A' and columns only start with 'B'
    cmap = cmap[cmap.index.str.startswith('A')]
    cmap = cmap[cmap.columns[cmap.columns.str.startswith('B')]]

    fig = go.Figure(go.Heatmap(z=cmap.values,
                                x=cmap.columns,
                                y=cmap.index,
                            colorscale='gray',
                            reversescale=True))
    fig.update_layout(
        width=1000,
        height=1000)
    fig.write_html('data/heatmap2.html')

def plot_cmap(cmap):
    fig = go.Figure(go.Heatmap(z=cmap.values,
                            x=cmap.columns,
                            y=cmap.index,
                           colorscale='gray',
                           reversescale=True))
    fig.update_layout(
        width=1000,
        height=1000)
    fig.show()

def get_cmap(pdb_filename, dist=8):
    c = contactMap(pdb_filename, dist=dist)['data']
    root_ids = {f"{item['root']['chainID']}:{item['root']['resID']}" for item in c}
    partner_ids = {f"{item['partners'][i]['chainID']}:{item['partners'][i]['resID']}" for item in c for i in range(len(item['partners']))}
    ids = list(root_ids.union(partner_ids))
    # sort ids: first all IDs that start with A, then all IDs that start with B.
    # after A/B comes a : and then a number. Sort by the number.
    ids = sorted(ids, key=lambda x: (x[0], int(x.split(':')[1])))
    cmap = pd.DataFrame(np.zeros((len(ids), len(ids)), np.float64))
    cmap.index = ids
    cmap.columns = ids
    for i in range(len(c)):
        row_id = c[i]['root']['chainID'] + ":" + c[i]['root']['resID']
        for j in range(len(c[i]['partners'])):
            row_id2 = c[i]['partners'][j]['chainID'] + ":" + c[i]['partners'][j]['resID']
            cmap.at[row_id, row_id2] = 1
            cmap.at[row_id2, row_id] = 1

    # filter such that index only starts with 'A' and columns only start with 'B'
    cmap = cmap[cmap.index.str.startswith('A')]
    cmap = cmap[cmap.columns[cmap.columns.str.startswith('B')]]
    return cmap

def get_distmap(pdb_code, pdb_filename):
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
    model = structure[0]
    dist_matrix = calc_dist_matrix(model["A"], model["B"])
    x = [f"{str(res.resname)}:{str(res.id[1])}" for res in model["B"]]
    y = [f"{str(res.resname)}:{str(res.id[1])}" for res in model["A"]]
    return dist_matrix, x, y

def plot_distmap(distmap, x, y):
    fig = go.Figure(go.Heatmap(z=distmap.astype(np.float64),
                        x=x,
                        y=y,
                        colorscale='blues',
                        reversescale=True))
    fig.update_layout(
        width=1000,
        height=1000)
    fig.show()


def plot_distmaps(distmap1, distmap2, x0, y0, x1, y1, complex_id, id1, id2, model):
    model_name_mapping = {
        "dscript_like": "DSCRIPT-like",
        "baseline2d": "2d-baseline",
        "selfattention": "Selfattention",
        "crossattention": "Crossattention"
    }
    model_name = model_name_mapping[model]
    distmap1_flat = distmap1.flatten()
    distmap2_flat = distmap2.flatten()
    correlation = np.corrcoef(distmap1_flat, distmap2_flat)[0, 1]

    fig = make_subplots(rows=1, cols=2)

    # real
    fig.add_trace(
        go.Heatmap(z=distmap1.astype(np.float64),
                   x=x0,
                   y=y0,
                   colorscale='blues',
                   reversescale=True,
                   showscale=False),
        row=1, col=1)

    # predicted
    fig.add_trace(
        go.Heatmap(z=distmap2.astype(np.float64),
                   x=x1,
                   y=y1,
                   colorscale='blues',
                   reversescale=True,
                   showscale=False),
        row=1, col=2)

    # correlation
    fig.add_annotation(
        x=0.5,
        y=-0.12,
        xref='paper',
        yref='paper',
        text=f'Correlation: {correlation:.2f}',
        showarrow=False,
        font=dict(
            size=40,
            color="red"
        ))
    
    fig.update_xaxes(title_text="Real", title_font=dict(size=32), row=1, col=1)
    fig.update_xaxes(title_text="Predicted", title_font=dict(size=32), row=1, col=2)

    fig.update_layout(
        title=f"{model_name}: Complex ID: {complex_id}, IDs: {id1}, {id2}",
        title_x=0.5,
        title_font=dict(size=40,
                        color="black"),
        width=2000,
        height=1000)

    #fig.show()
    pio.write_image(fig, f"/nfs/home/students/t.reim/bachelor/pytorchtest/data/distmaps/{model_name}_{complex_id}_{id1}_{id2}.png")

    return correlation
