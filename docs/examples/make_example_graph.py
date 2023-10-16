""" Illustrates the dependency graph between the examples as a tree. """
import argparse
import sys
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

_examples_folder = Path("docs/examples")
pytorch_setup = Path("frameworks/pytorch_setup")
single_gpu = Path("distributed/single_gpu")
multi_gpu = Path("distributed/multi_gpu")
multi_node = Path("distributed/multi_node")
checkpointing = Path("good_practices/checkpointing")
wandb = Path("good_practices/wandb_setup")
orion = Path("good_practices/hpo_with_orion")

G = nx.DiGraph()
G.add_edges_from(
    [
        (pytorch_setup, single_gpu),
        (single_gpu, multi_gpu),
        (multi_gpu, multi_node),
        (single_gpu, checkpointing),
        (single_gpu, checkpointing),
    ]
)

# example_dependencies: dict[Path, Path] = {
#     single_gpu: pytorch_setup,
#     multi_gpu: single_gpu,
#     multi_node: multi_gpu,
#     checkpointing: single_gpu,
# }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "example_dir", type=Path, help="The current node in the example directory."
    )
    parser.add_argument(
        "output_path", type=Path, help="Output path where the figure should be saved."
    )
    args = parser.parse_args()
    example_dir: Path = args.example_dir
    output_path: Path = args.output_path

    show_graph(example_dir, out_path=output_path)


def show_graph(current_example: Path, out_path: Path):
    all_nodes = list(G.nodes())

    req_edges: list[tuple[Path, Path]] = []
    req_nodes: list[Path] = []

    req_edges: list[tuple[Path, Path]] = []
    for prereq_example in nx.ancestors(G, current_example):
        req_nodes.append(prereq_example)
        print(f"{prereq_example} is a prerequisite for {current_example}")
        for path in nx.all_simple_paths(G, prereq_example, current_example):
            for edge1, edge2 in zip(path[:-1], path[1:]):
                req_edges.append((edge1, edge2))

    # for example, example_dep in example_dependencies.items():
    #     G.add_edge(example_dep, example)

    # for node in G.nodes():
    #     G.attr("node", URL="https://stackoverflow.com")

    # val_map = {}  # {single_gpu: 1.0, checkpointing: 0.5714285714285714, multi_node: 0.0}
    # values = [val_map.get(node, 0.25) for node in G.nodes()]

    print(req_edges)
    # edge_colours = ["red" if edge in req_edges else "black" for edge in G.edges()]
    # node_colours = ["red" if node in req_nodes else "black" for node in G.nodes()]
    unreq_edges = [edge for edge in G.edges() if edge not in req_edges]
    unreq_nodes = [node for node in G.nodes() if node not in req_nodes]
    # black_edges = [edge for edge in G.edges() if edge not in req_edges]

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    # pos = nx.spring_layout(G)
    # pos = nx.planar_layout(G)

    # Show the current example node in red, and the other nodes in black.

    # write dot file to use with graphviz
    # run "dot -Tpng test.dot >test.png"
    # nx.nx_agraph.write_dot(G, "test.dot")
    pos = graphviz_layout(G, prog="dot")
    # nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap("jet"), node_color=values, node_size=0)
    # nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_labels(G, pos, alpha=0.5)
    required_nodes_graph = nx.Graph()
    required_nodes_graph.add_nodes_from(req_nodes)
    nx.draw_networkx_labels(required_nodes_graph, pos, font_color="black")

    unrequired_nodes_graph = nx.Graph()
    unrequired_nodes_graph.add_nodes_from(unreq_nodes)
    nx.draw_networkx_labels(unrequired_nodes_graph, pos, font_color="gray")

    current_node_graph = nx.Graph()
    current_node_graph.add_node(current_example)
    bbox = {"ec": "k", "fc": "white", "alpha": 0.7}
    nx.draw_networkx_labels(current_node_graph, pos, font_color="blue", bbox=bbox)
    arrow_margin = 30
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=req_edges,
        edge_color="blue",
        arrows=True,
        min_source_margin=arrow_margin,
        min_target_margin=arrow_margin,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=unreq_edges,
        edge_color="gray",
        arrows=True,
        min_source_margin=arrow_margin,
        min_target_margin=arrow_margin,
    )
    # TODO: Trying to add some links in the image, alas it doesnt' work.
    # axis = plt.gca()
    # axis.set_url(["https://www.bbc.com/news", "https://www.google.com/", None])
    plt.savefig(out_path)
    plt.show()


if __name__ == "__main__":
    main()
