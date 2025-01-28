import os
from datetime import timedelta

import networkx as nx
import pandas as pd
from datetime import timedelta
import networkx as nx
from datetime import timedelta
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as animation
# P 8.6, 8.7
def parse_gtfs(path, service_id=None, first_trip_only=False):
    """
    Creates a directed graph from GTFS data where nodes represent stops and edges represent connections.

    The function reads GTFS files (stops.txt, routes.txt, trips.txt, stop_times.txt) and creates a networkx
    DiGraph where:
    - Nodes represent stops and are identified by their stop_id with attributes:
        * lines (set): routes serving the stop
        * types (set): transport types serving the stop (e.g., 'bus', 'tram', 'rail')
        * lat (float): latitude coordinate of the stop
        * lng (float): longitude coordinate of the stop
        * name (str): name of the stop
        * code (str): stop code (if available, else None)
    - Edges represent connections between consecutive stops on routes with attributes:
        * lines (set): routes using the connection
        * types (set): transport types using the connection
        * distance (float): distance between stops in kilometers
        * vehicles (list): list of dictionaries containing:
            - departure (timedelta): departure time from source stop
            - arrival (timedelta): arrival time at target stop (minus 1 second)
            - line (str): route identifier

    Args:
        path (str): Path to directory containing GTFS files
        service_id (str, optional): Filter trips by service_id. If None, all trips are included.
        first_trip_only (bool): If True, only use the first trip for each route and direction.
            This means up to two trips per route: one for direction_id=0 and one for direction_id=1.
            Default: False

    Returns:
        tuple[nx.DiGraph, dict]: A tuple containing:
            - nx.DiGraph: Directed graph representing the transport network
            - dict: Mapping of route_id to transport type (e.g., 'bus', 'tram', etc.)
    """
    G = nx.DiGraph()

    stops_df = pd.read_csv(os.path.join(path, "stops.txt"))
    routes_df = pd.read_csv(os.path.join(path, "routes.txt"))
    trips_df = pd.read_csv(os.path.join(path, "trips.txt"))
    times_df = pd.read_csv(os.path.join(path, "stop_times.txt"))

    type_mapping = {
        0: "tram",
        1: "subway",
        2: "rail",
        3: "bus",
        4: "ferry",
        5: "cable_tram",
        6: "aerial_lift",
        7: "funicular",
    }
    route_types = {
        route_id: type_mapping.get(route_type, "unknown")
        for route_id, route_type in zip(routes_df.route_id, routes_df.route_type)
    }

    print("Adding nodes...")
    nodes_data = [
        (
            stop.stop_id,
            {
                "lat": float(stop.stop_lat),
                "lng": float(stop.stop_lon),
                "lines": set(),
                "types": set(),
                "name": stop.stop_name,
                "code": (
                    stop.stop_code
                    if "stop_code" in stop and pd.notna(stop.stop_code)
                    else None
                ),
            },
        )
        for _, stop in stops_df.iterrows()
    ]
    G.add_nodes_from(nodes_data)

    # Filter trips
    if service_id is not None:
        trips_df = trips_df[trips_df.service_id == service_id]
    if first_trip_only:
        trips_df = trips_df.groupby(["route_id", "direction_id"]).first().reset_index()

    # Add route_id to stop times data
    times_df = pd.merge(
        times_df,
        trips_df[["trip_id", "route_id"]],
        on="trip_id",
        how="inner",
    )

    edges_data = {}
    all_trips_count = len(times_df.trip_id.unique())
    processed_trips_count = 0

    for trip_id, group in times_df.groupby("trip_id"):
        route_id = group.iloc[0].route_id
        route_type = route_types[route_id]

        processed_trips_count += 1
        print(
            f"Processing trip {trip_id}, route_id {route_id}, route_type {route_type} ({processed_trips_count}/{all_trips_count})"
        )

        stops = group.sort_values("stop_sequence")

        # Update node attributes
        for stop_id in stops.stop_id:
            G.nodes[stop_id]["lines"].add(route_id)
            G.nodes[stop_id]["types"].add(route_type)

        # Create edges
        for i in range(len(stops) - 1):
            row1, row2 = stops.iloc[i], stops.iloc[i + 1]
            stop1, stop2 = row1.stop_id, row2.stop_id

            dist = float(row2.shape_dist_traveled) - float(row1.shape_dist_traveled)
            vehicle_info = {
                "departure": parse_time(row1.departure_time),
                "arrival": parse_time(row2.arrival_time) - timedelta(seconds=1),
                "line": route_id,
            }

            if (stop1, stop2) in edges_data:
                edges_data[(stop1, stop2)]["lines"].add(route_id)
                edges_data[(stop1, stop2)]["types"].add(route_type)
                edges_data[(stop1, stop2)]["vehicles"].append(vehicle_info)
            else:
                edges_data[(stop1, stop2)] = {
                    "lines": {route_id},
                    "types": {route_type},
                    "distance": dist,
                    "vehicles": [vehicle_info],
                }

    print("Adding edges...")
    edges = [(stop1, stop2, data) for (stop1, stop2), data in edges_data.items()]
    G.add_edges_from(edges)

    return G, route_types

def filter_graph_by_transport_type(G, transport_type):
    G_filtered = G.copy()
    nodes_to_remove = [node for node, data in G_filtered.nodes(data=True) if transport_type not in data['types']]
    G_filtered.remove_nodes_from(nodes_to_remove)
    return G_filtered

def parse_time(time_str: str) -> timedelta:
    h, m, s = map(int, time_str.split(":"))
    return timedelta(hours=h % 24, minutes=m, seconds=s) + timedelta(days=h // 24)


def create_temporal_snapshots(G, start_time, end_time, interval_minutes):
    """
    Create temporal snapshots of the graph based on active connections within time intervals.

    Args:
        G (nx.DiGraph): The original transport graph with time data on edges.
        start_time (str): Start time in "HH:MM:SS" format.
        end_time (str): End time in "HH:MM:SS" format.
        interval_minutes (int): Time interval in minutes.

    Returns:
        list[tuple[str, nx.DiGraph]]: List of tuples where each tuple contains a time range and a subgraph.
    """
    snapshots = []
    start_dt = parse_time(start_time)
    end_dt = parse_time(end_time)
    current_dt = start_dt

    while current_dt < end_dt:
        next_dt = current_dt + timedelta(minutes=interval_minutes)
        subgraph = nx.DiGraph()

        for u, v, data in G.edges(data=True):
            active_vehicles = [
                v for v in data["vehicles"]
                if current_dt <= v["departure"] < next_dt
            ]
            if active_vehicles:
                subgraph.add_edge(u, v, **data)

        snapshots.append((f"{str(current_dt)} - {str(next_dt)}", subgraph))
        current_dt = next_dt

    return snapshots


def animate_traffic(snapshots, pos_geo):
    """
    Create an animation showing traffic flow throughout the day.

    Args:
        snapshots (list[tuple[str, nx.DiGraph]]): Temporal snapshots of the network.
        pos_geo (dict): Geographic positions of nodes.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    node_color = "blue"
    edge_color = "gray"
    node_size = 10

    def update(frame):
        ax.clear()
        time_range, G_snapshot = snapshots[frame]
        ax.set_title(f"Traffic Flow: {time_range}")
        nx.draw(
            G_snapshot,
            pos_geo,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            with_labels=False,
            ax=ax
        )

    ani = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=500, repeat=True
    )
    plt.show()

import numpy as np

def interpolate_positions(snapshots, pos_geo):
    """
    Interpolates node positions for smooth transitions between snapshots.

    Args:
        snapshots (list[tuple[str, nx.DiGraph]]): Temporal snapshots of the network.
        pos_geo (dict): Geographic positions of nodes.

    Returns:
        list[dict]: List of interpolated positions for each frame.
    """
    interpolated_positions = []
    num_frames = len(snapshots)

    for frame in range(num_frames):
        _, G_snapshot = snapshots[frame]
        positions = {node: pos_geo.get(node, (0, 0)) for node in G_snapshot.nodes}

        if frame > 0:
            prev_positions = interpolated_positions[-1]
            for node in positions:
                if node in prev_positions:
                    # Interpolate between previous and current positions
                    positions[node] = tuple(np.mean([prev_positions[node], positions[node]], axis=0))

        interpolated_positions.append(positions)

    return interpolated_positions


def animate_traffic_smooth(snapshots, pos_geo):
    """
    Create a smooth animation showing traffic flow throughout the day.

    Args:
        snapshots (list[tuple[str, nx.DiGraph]]): Temporal snapshots of the network.
        pos_geo (dict): Geographic positions of nodes.
    """
    interpolated_positions = interpolate_positions(snapshots, pos_geo)

    fig, ax = plt.subplots(figsize=(8, 8))
    node_color = "blue"
    edge_color = "gray"
    node_size = 10

    def update(frame):
        ax.clear()
        time_range, G_snapshot = snapshots[frame]
        ax.set_title(f"Traffic Flow: {time_range}")
        positions = interpolated_positions[frame]
        nx.draw(
            G_snapshot,
            positions,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            with_labels=False,
            ax=ax
        )

    ani = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=500, repeat=True
    )
    plt.show()


start_time = "09:47:00"
end_time = "11:47:00"
interval_minutes = 1


G, route_types = parse_gtfs("lab8/gtfs", service_id="3_2", first_trip_only=True)
G_bus = filter_graph_by_transport_type(G, "bus")
pos_geo = {node: (data['lng'], data['lat']) for node, data in G_bus.nodes(data=True)}
snapshots = create_temporal_snapshots(G_bus, start_time, end_time, interval_minutes)

animate_traffic_smooth(snapshots, pos_geo)