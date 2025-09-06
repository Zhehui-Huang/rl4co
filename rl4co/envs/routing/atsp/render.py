import matplotlib.pyplot as plt
import numpy as np
import torch

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    if ax is None:
        # Create a plot of the nodes
        _, ax = plt.subplots()

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)

    # If batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    # ATSP environments use cost matrices, not coordinates
    # Check if coordinates are available for visualization
    if "locs" in td:
        locs = td["locs"]
    else:
        # Generate coordinates using multi-dimensional scaling from cost matrix
        cost_matrix = td["cost_matrix"]
        num_nodes = cost_matrix.size(0)
        
        # Use classical MDS to convert distance matrix to 2D coordinates
        try:
            from sklearn.manifold import MDS
        except ImportError:
            log.warning("sklearn not available, using random coordinates for ATSP visualization")
            # Fallback to random coordinates if sklearn is not available
            locs = torch.rand(num_nodes, 2)
        else:
            # Convert cost matrix to numpy for MDS
            cost_np = cost_matrix.numpy()
            
            # Ensure the cost matrix is symmetric for MDS
            # ATSP cost matrices are generally asymmetric, so we symmetrize
            cost_symmetric = (cost_np + cost_np.T) / 2
            
            # Apply MDS to generate 2D coordinates
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords = mds.fit_transform(cost_symmetric)
            
            # Convert back to torch tensor
            locs = torch.from_numpy(coords).float()
            
            log.info("ATSP environment: Generated coordinates using MDS from cost matrix for visualization.")

    # Gather locs in order of action if available
    if actions is None:
        log.warning("No action in TensorDict, rendering unsorted locs")
    else:
        actions = actions.detach().cpu()
        locs = gather_by_index(locs, actions, dim=0)

    # Cat the first node to the end to complete the tour
    locs = torch.cat((locs, locs[0:1]))
    x, y = locs[:, 0], locs[:, 1]

    # Plot the visited nodes
    ax.scatter(x, y, color="tab:blue")

    # Add arrows between visited nodes as a quiver plot
    dx, dy = np.diff(x), np.diff(y)
    ax.quiver(x[:-1], y[:-1], dx, dy, scale_units="xy", angles="xy", scale=1, color="k")
    
    # Add node labels to show the sequence
    for i, (xi, yi) in enumerate(zip(x[:-1], y[:-1])):
        ax.annotate(str(i), (xi, yi), xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, ha='left')

    ax.set_title("ATSP Tour (visualization with dummy coordinates)")
    ax.set_aspect('equal')
    
    return ax
