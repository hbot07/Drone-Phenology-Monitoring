
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tree_tracking import TreeTrackingGraph

app = Flask(__name__)

# Global tracking graph instance (reload on demand)
tracking_graph = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_tracking', methods=['POST'])
def run_tracking():
    global tracking_graph
    # Optionally allow user to set parameters via form
    tracking_graph = TreeTrackingGraph(
        crown_dir='../input_crowns',
        ortho_dir='../input_om',
        iou_threshold=0.15,
        resize_factor=0.1,
        simplify_tol=1.0,
        max_crowns=200
    )
    tracking_graph.process_all_hungarian()
    report, metrics = tracking_graph.quality_report()
    return jsonify({'report': report, 'metrics': metrics})

@app.route('/metrics')
def metrics():
    global tracking_graph
    if tracking_graph is None:
        return jsonify({'error': 'Tracking not run yet'})
    report, metrics = tracking_graph.quality_report()
    return jsonify({'report': report, 'metrics': metrics})

@app.route('/visualize_chain/<int:om_id>/<int:crown_id>')
def visualize_chain(om_id, crown_id):
    global tracking_graph
    if tracking_graph is None:
        return jsonify({'error': 'Tracking not run yet'})
    node = (om_id, crown_id)
    if node not in tracking_graph.G.nodes:
        return jsonify({'error': f'Node {node} not found'})
    chain = tracking_graph.get_matching_chain(node)
    fig = tracking_graph.plot_matching_chain_plotly(chain)
    return jsonify({'fig': fig.to_json()})

@app.route('/orthomosaic_with_crowns')
def orthomosaic_with_crowns():
    global tracking_graph
    if tracking_graph is None:
        return jsonify({'error': 'Tracking not run yet'})
    fig = tracking_graph.plot_orthomosaic_with_crowns_plotly(om_idx=0)
    if fig is None:
        return jsonify({'error': 'No orthomosaic found'})
    return jsonify({'fig': fig.to_json()})

if __name__ == '__main__':
    app.run(debug=True)
