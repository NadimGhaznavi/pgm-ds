from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from power_grid_model_ds import Grid
from power_grid_model_ds.arrays import LineArray, NodeArray
from power_grid_model_ds.generators import RadialGridGenerator
from power_grid_model import LoadGenType
from power_grid_model_ds.arrays import SymLoadArray
from power_grid_model_ds.enums import NodeType
from power_grid_model_ds import PowerGridModelInterface
from power_grid_model_ds.visualizer import visualize

R_PER_KM = 0.1
X_PER_KM = 0.1

RANDOM_GENERATOR_SEED = 1970


class ExtendedNodeArray(NodeArray):
    """Extends the node array with the simulated voltage and coordinates"""

    _defaults = {"u": 0}

    u: NDArray[np.float64]
    x_coor: NDArray[np.float64]
    y_coor: NDArray[np.float64]

    @property
    def is_overloaded(self):
        return np.logical_or(self.u > 1.1 * self.u_rated, self.u < 0.9 * self.u_rated)


class ExtendedLineArray(LineArray):
    """Extends the line array with current output"""

    _defaults = {"i_from": 0}

    i_from: NDArray[np.float64]

    @property
    def is_overloaded(self):
        return self.i_from > self.i_n


@dataclass
class ExtendedGrid(Grid):
    node: ExtendedNodeArray
    line: ExtendedLineArray


def create_new_consumer_arrays(
    u_rated: float, x_coor: float, y_coor: float, p_specified: float, q_specified: float
) -> tuple[ExtendedNodeArray, SymLoadArray]:
    new_consumer = ExtendedNodeArray(
        u_rated=[u_rated],
        node_type=[NodeType.UNSPECIFIED],
        x_coor=[x_coor],
        y_coor=[y_coor],
    )
    new_consumer_load = SymLoadArray(
        node=[new_consumer.get_empty_value("id")],
        status=[1],
        type=[LoadGenType.const_power],
        p_specified=[p_specified],
        q_specified=[q_specified],
    )
    return new_consumer, new_consumer_load


def find_closest_node(grid: ExtendedGrid, x: float, y: float) -> int:
    dist = np.sqrt((grid.node.x_coor - x) ** 2 + (grid.node.y_coor - y) ** 2)
    return np.argmin(dist).item()


def connect_new_consumer(
    grid: ExtendedGrid,
    new_consumer: ExtendedNodeArray,
    new_consumer_load: SymLoadArray,
):
    closest_node_idx = find_closest_node(
        grid=grid,
        x=new_consumer.x_coor[0],
        y=new_consumer.y_coor[0],
    )
    closest_node = grid.node[closest_node_idx]

    grid.append(new_consumer)
    new_consumer_load.node = new_consumer.id
    grid.append(new_consumer_load)

    dist = np.sqrt(
        (closest_node.x_coor - new_consumer.x_coor) ** 2
        + (closest_node.y_coor - new_consumer.y_coor) ** 2
    )

    new_line = ExtendedLineArray(
        from_node=[closest_node.id],
        to_node=[new_consumer.id],
        from_status=[1],
        to_status=[1],
        r1=[R_PER_KM * dist / 1_000],
        x1=[X_PER_KM * dist / 1_000],
        c1=[0],
        tan1=[0],
        i_n=[200],
    )
    grid.append(new_line)


def update_grid(grid: ExtendedGrid):
    # Set the new feeder ids
    grid.set_feeder_ids()

    # Update the power flow
    core_interface = PowerGridModelInterface(grid=grid)

    core_interface.create_input_from_grid()
    core_interface.calculate_power_flow()
    core_interface.update_grid()


import rustworkx as rx
import matplotlib.pyplot as plt


def main():
    # Create a random grid generator
    grid_generator = RadialGridGenerator(
        grid_class=ExtendedGrid,  # Create a custom grid
        nr_nodes=20,  # Create 20 regular grid nodes
        nr_sources=1,  # Create a single source of power
        nr_nops=10,  # Create 10 'normally open points'
    )

    # Create a grid using the generator
    grid = grid_generator.run(seed=RANDOM_GENERATOR_SEED)

    grid.set_feeder_ids()

    rng = np.random.default_rng()

    grid.node.x_coor = rng.uniform(100, 500, len(grid.node))
    grid.node.y_coor = rng.uniform(100, 500, len(grid.node))

    new_consumer, new_consumer_load = create_new_consumer_arrays(
        u_rated=10_500,
        x_coor=300,
        y_coor=300,
        p_specified=1_000_000,
        q_specified=200_000,
    )

    connect_new_consumer(grid, new_consumer, new_consumer_load)
    update_grid(grid)

    print("\nNodes:")
    # print(grid.node)
    for node in grid.node:
        print(f"{node}\n")

    print("\nLines:")
    # print(grid.line)
    for line in grid.line:
        print(f"{line}\n")

    # print("\nOverloaded nodes:")
    # print(grid.node[grid.node.is_overloaded])

    # print("\nOverloaded lines:")
    # print(grid.line[grid.line.is_overloaded])

    visualize(grid)


if __name__ == "__main__":
    main()
