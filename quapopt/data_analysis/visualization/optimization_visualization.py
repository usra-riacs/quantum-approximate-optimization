# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
# Use, duplication, or disclosure without authors' permission is strictly prohibited.
 
from typing import List, Tuple

import numpy as np
import pandas as pd

from plotly.subplots import make_subplots

import plotly.graph_objects as go
from scipy.interpolate import griddata
from quapopt.data_analysis.data_handling import STANDARD_NAMES_DATA_TYPES as SNDT, STANDARD_NAMES_VARIABLES as SNV


_PLOTTING_NICE_NAMES = {f"{SNV.Angles.id_long}-0":"γ",
                        f"{SNV.Angles.id_long}-1":"β",
                        f"{SNV.EnergyMean.id_long}": "<E>",
                        f"{SNV.EnergySTD.id_long}": "σ(E)",
                        f"{SNV.EnergyBest.id_long}": "E",
                        f"{SNV.ApproximationRatioMean.id_long}":"<AR>",
                        f"{SNV.ApproximationRatioSTD.id_long}":"σ(AR)",
                        f"{SNV.ApproximationRatioBest.id_long}":"AR",
                        }
def get_nice_variable_name(name:str):
    if name in _PLOTTING_NICE_NAMES.keys():
        return _PLOTTING_NICE_NAMES[name]
    return name
_gnvn = get_nice_variable_name


def get_interpolated_heatmap(df: pd.DataFrame,
                                x_name: str,
                                y_name: str,
                                fom_name: str,
                                bounds_x: Tuple[float, float]=None,
                                bounds_y: Tuple[float, float]=None,
                                min_value=None,
                                max_value=None,
                                grid_resolution=1000,
                             in_3d=False,
                             colormap_name='Viridis_r') -> Tuple[go.Heatmap, go.Scatter]:

    #print("HEJ:",in_3d)
    if bounds_x is None:
        bounds_x = (df[x_name].min(), df[x_name].max())
    if bounds_y is None:
        bounds_y = (df[y_name].min(), df[y_name].max())

    # grid_resolution = 1000  # Changeable resolution
    # Create a fixed grid
    grid_x, grid_y = np.linspace(*bounds_x, grid_resolution), np.linspace(*bounds_y, grid_resolution)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Interpolate values onto the fixed grid
    grid_values = griddata(
        (df[x_name], df[y_name]),  # Points from the optimizer's trajectory
        df[fom_name],  # Values at those points
        (grid_x, grid_y),  # Target grid for interpolation
        method='cubic'
    )
    # Create the heatmap
    #colorscale_heatmap = 'Viridis_r'
    colorbar_heatmap = dict(title=f"{_gnvn(fom_name)}", xpad=5.0, ypad=50.0)
    markersize_scatter = 0.25


    scatter_markers_dict = dict(
                        color='black',
                        size=markersize_scatter,
                        symbol='circle',
                        line=dict(color="rgba(255, 255, 255, 0.5)", width=1)
                    )
    scatter_hovertemplate = (f"{_gnvn(x_name)}: %{{x}}"
                             f"<br>{_gnvn(y_name)}: %{{y}}"
                             f"<br>{_gnvn(fom_name)}: %{{text}}<extra></extra>")


    if in_3d:
        heatmap = go.Surface(
            z=grid_values,
            x=np.linspace(*bounds_x, grid_resolution),
            y=np.linspace(*bounds_y, grid_resolution),
            colorscale=colormap_name,
            colorbar=colorbar_heatmap,
            cmin=min_value,
            cmax=max_value,
        )
        scatter = go.Scatter3d(
            x=df[x_name],
            y=df[y_name],
            z=df[fom_name],
            mode='markers',
            marker=scatter_markers_dict,
            text=df[fom_name],
            hovertemplate=scatter_hovertemplate,
        )

    else:
        heatmap = go.Heatmap(
            z=grid_values,
            x=np.linspace(*bounds_x, grid_resolution),
            y=np.linspace(*bounds_y, grid_resolution),
            colorscale=colormap_name,
            colorbar=colorbar_heatmap,
            zmin=min_value,
            zmax=max_value,
            zorder=0
        )

        # Overlay the white background where NaNs are present
        heatmap['z'] = np.where(np.isnan(grid_values), None, grid_values)

        scatter = go.Scatter(
                        x=df[x_name],
                        y=df[y_name],
                        mode='markers',
                        marker=scatter_markers_dict,
                        text=df[fom_name],
                        hovertemplate=scatter_hovertemplate,
                        zorder=1
                    )

    return heatmap, scatter


def get_optimization_trajectory(df: pd.DataFrame,
                                   x_name: str,
                                   y_name: str,
                                   fom_name: str,
                                   names_suffix: str = '',
                                   minimization=True,
                                    in_3d=False) -> Tuple[go.Scatter, go.Scatter, go.Scatter]:

    if in_3d:
        hover_z = fom_name + ": %{z}<extra></extra>"
    else:
        hover_z = fom_name + ": %{customdata}<extra></extra>"


    hovertemplate = (
            f"{_gnvn(x_name)}" + ": %{x}<br>" +
            f"{_gnvn(y_name)}" + ": %{y}<br>" +
            hover_z
    )

    path_name = f"Path {names_suffix}"
    start_name = f"Start {names_suffix}"
    best_name = f"Best {names_suffix}"

    # find THE BEST point
    if minimization:
        best_point = df.loc[df[fom_name].idxmin()]
    else:
        best_point = df.loc[df[fom_name].idxmax()]
    if in_3d:
        dash_style = 'solid'
    else:
        dash_style = 'dot'

    line_dict_main = dict(color='red', width=0.5, dash=dash_style)

    markersize_scatter = 4

    if in_3d:
        markersize_scatter*=0.5

    marker_dict_main = dict(size=markersize_scatter, color='red')
    marker_dict_start = dict(size=2*markersize_scatter, color='blue', symbol='x')
    marker_dict_best = dict(size=2*markersize_scatter, color='magenta', symbol='diamond')

    if in_3d:

        trajectory = go.Scatter3d(
            x=df[x_name],
            y=df[y_name],
            z=df[fom_name],
            mode='lines+markers',
            line=line_dict_main,
            marker=marker_dict_main,
            name=path_name,
            hovertemplate=hovertemplate,
        )

        trajectory_first = go.Scatter3d(
            x=[df[x_name].iloc[0]],
            y=[df[y_name].iloc[0]],
            z=[df[fom_name].iloc[0]],
            mode='markers',
            marker=marker_dict_start,
            hovertemplate=hovertemplate,
            name=start_name,
        )
        trajectory_best = go.Scatter3d(
            x=[best_point[x_name]],
            y=[best_point[y_name]],
            z=[best_point[fom_name]],
            mode='markers',
            marker=marker_dict_best,
            name=best_name,
            hovertemplate=hovertemplate,
        )


    else:

        # Add the optimizer's trajectory
        trajectory = go.Scatter(
            x=df[x_name],
            y=df[y_name],
            mode='lines+markers',
            line=line_dict_main,
            marker=marker_dict_main,
            name=path_name,
            customdata=df[fom_name],
            hovertemplate=hovertemplate,
            zorder=2
        )

        # add first point with special symbol
        trajectory_first = go.Scatter(
            x=[df[x_name].iloc[0]],
            y=[df[y_name].iloc[0]],
            mode='markers',
            marker=marker_dict_start,
            customdata=[df[fom_name].iloc[0]],
            hovertemplate=hovertemplate,
            name=start_name,
            zorder=3
        )

        # add best point with special symbol
        trajectory_best = go.Scatter(
            x=[best_point[x_name]],
            y=[best_point[y_name]],
            mode='markers',
            marker=marker_dict_best,

            name=best_name,
            customdata=[best_point[fom_name]],
            hovertemplate=hovertemplate,
            zorder = 3,
        )

    return trajectory, trajectory_first, trajectory_best

"""
    Some small modification
"""
def get_simple_optimization_trajectory(df: pd.DataFrame,
                                       x_name: str,
                                       y_name: str,
                                       fom_name: str,
                                       names_suffix: str = '',
                                       in_3d:bool=False) -> Tuple[go.Scatter, go.Scatter, go.Scatter]:
    path_name = f"Path {names_suffix}"
    start_name = f"Start {names_suffix}"
    best_name = f"Best {names_suffix}"

    # find THE BEST point   
    dash_style = 'dot'
    line_dict_main = dict(color='red', width=0.5, dash=dash_style)
    markersize_scatter = 4

    marker_dict_main = dict(size=markersize_scatter, color='red')
    marker_dict_start = dict(size=2*markersize_scatter, color='blue', symbol='x')
    marker_dict_best = dict(size=2*markersize_scatter, color='magenta', symbol='diamond')
    # Add the optimizer's trajectory

    if in_3d:
        trajectory = go.Scatter3d(
                        x=df[x_name],
                        y=df[y_name],
                        z=df[fom_name],
                        hovertemplate=(f"{_gnvn(x_name)}: %{{x}}<br>",
                                       f"{_gnvn(y_name)}: %{{y}}<br>",
                                       f"{_gnvn(fom_name)}: %{{z}}<extra></extra>"),
                        mode="lines+markers",
                        line=line_dict_main,
                        marker=marker_dict_main,
                        name=path_name
                    )

    else:
        trajectory = go.Scatter(
            x=df[x_name],
            y=df[y_name],
            #add also hover info about function values:
            customdata=df[fom_name],
            hovertemplate=(f"{_gnvn(x_name)}: %{{x}}<br>"
                           f"{_gnvn(y_name)}: %{{y}}<br>"
                           f"{_gnvn(fom_name)}: %{{customdata}}<extra></extra>"),
            mode='lines+markers',
            line=line_dict_main,
            marker=marker_dict_main,
            name=path_name,
            zorder=2
        )

    # add first point with special symbol
    # if fom_name in df.columns:
    #     print(f"First point of trajectory: {(df[x_name].iloc[0], df[y_name].iloc[0])} with value {df[fom_name].iloc[0]}")


    if in_3d:
        trajectory_first = go.Scatter3d(
            x=[df[x_name].iloc[0]],
            y=[df[y_name].iloc[0]],
            z=[df[fom_name].iloc[0]],
            mode='markers',
            marker=marker_dict_start,
            name=start_name
        )
    else:
        trajectory_first = go.Scatter(
            x=[df[x_name].iloc[0]],
            y=[df[y_name].iloc[0]],
            mode='markers',
            marker=marker_dict_start,
            name=start_name,
            zorder=3
        )
    if fom_name in df.columns:
        # add best point with special symbol
        best_point = df.loc[df[fom_name].idxmin()]
        #print(f"Best point of trajectory: {(best_point[x_name], best_point[y_name])} with value {best_point[fom_name]}")

        if in_3d:
            trajectory_best = go.Scatter3d(
                x=[best_point[x_name]],
                y=[best_point[y_name]],
                z=[best_point[fom_name]],
                mode='markers',
                marker=marker_dict_best,
                name=best_name,
            )

        else:
            trajectory_best = go.Scatter(
                x=[best_point[x_name]],
                y=[best_point[y_name]],
                mode='markers',
                marker=marker_dict_best,
                name=best_name,
                zorder=3,
            )


    else:
        # add best point with special symbol
        #print("Best point of trajectory: ", (df[x_name].iloc[-1], df[y_name].iloc[-1]))
        if in_3d:
            raise ValueError("In 3D, the fom_name must be present in the dataframe to determine the best point.")

        trajectory_best = go.Scatter(
            x=[df[x_name].iloc[-1]],
            y=[df[y_name].iloc[-1]],
            mode='markers',
            marker=marker_dict_best,
            name=best_name,
            zorder = 3,
        )
    return trajectory, trajectory_first, trajectory_best

def plot_interpolated_heatmap(df: pd.DataFrame,
                              x_name: str,
                              y_name: str,
                              fom_name: str,
                              bounds_x: Tuple[float, float]=None,
                              bounds_y: Tuple[float, float]=None,
                                 heatmap_input=None,
                                 scatter_input=None,
                                 add_trajectories=True,
                                 names_suffix='',
                                 title='2D heatmap',
                                 min_value=None,
                                 max_value=None,
                                 grid_resolution=1000,
                                 minimization=True,
                                 in_3d=False,
                              colormap_name='Viridis_r',
                              skip_heatmap:bool=False,
                              skip_scatter:bool=False) -> go.Figure:

    if skip_heatmap and skip_scatter:
        raise ValueError('Both skip_heatmap and skip_scatter are True. Nothing to plot.')

    all_data = []

    if heatmap_input is None or scatter_input is None:
        heatmap_input_possible, scatter_input_possible = get_interpolated_heatmap(df,
                                                                                  x_name=x_name,
                                                                                  y_name=y_name,
                                                                                     fom_name=fom_name,
                                                                                     bounds_x=bounds_x,
                                                                                     bounds_y=bounds_y,
                                                                                     min_value=min_value,
                                                                                     max_value=max_value,
                                                                                     grid_resolution=grid_resolution,
                                                                                  in_3d=in_3d,
                                                                                  colormap_name=colormap_name
                                                                                     )
        if scatter_input is None:
            scatter_input = scatter_input_possible
        if heatmap_input is None:
            heatmap_input = heatmap_input_possible



    if not skip_heatmap:
        all_data.append(heatmap_input)
    if not skip_scatter:
        all_data.append(scatter_input)

    #all_data = [heatmap_input,scatter_input]

    if add_trajectories:


        trajectory, trajectory_first, trajectory_last = get_simple_optimization_trajectory(df,
                                                                                           x_name=x_name,
                                                                                           y_name=y_name,
                                                                                           fom_name=fom_name,
                                                                                           names_suffix=names_suffix,
                                                                                           in_3d=in_3d)
        # Combine the heatmap and trajectory into a single plot
        all_data += [trajectory, trajectory_first, trajectory_last]

    fig = go.Figure(data=all_data)
    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=f"{_gnvn(x_name)}",
        yaxis_title=f"{_gnvn(y_name)}",
        xaxis=dict(range=bounds_x),
        yaxis=dict(range=bounds_y),
        template="plotly"
    )
    return fig



def plot_multiple_heatmaps(dataframes: List[pd.DataFrame],
                          x_name: str,
                          y_name: str,
                          fom_name: str,
                          bounds_x: Tuple[float, float]=None,
                          bounds_y: Tuple[float, float]=None,
                          heatmaps_input: List[go.Heatmap] = None,
                          scatter_inputs: List[go.Scatter] = None,
                          add_trajectories=True,
                          titles: List[str] = None,
                          suffixes: List[str] = None,
                          global_normalization: bool = True,
                          colormap_name='Viridis_r',
                          minimization=True,
                           in_3d=False,
                           skip_heatmaps=False,
                           skip_scatters=False) -> go.Figure:
    if skip_heatmaps and skip_scatters:
        raise ValueError('Both skip_heatmaps and skip_scatters are True. Nothing to plot.')


    if suffixes is None:
        if titles is None:
            suffixes = [''] * len(dataframes)
        else:
            suffixes = [f'({t})' for t in titles]
    if titles is None:
        titles = [''] * len(dataframes)
    if heatmaps_input is None:
        heatmaps_input = [None] * len(dataframes)

    if scatter_inputs is None:
        scatter_inputs = [None] * len(dataframes)

    # Create a subplot figure with 2 rows (one for each figure)
    specs = None
    number_of_rows = max([1, len(dataframes) // 2])
    if in_3d:
        specs = np.full((number_of_rows, 2), {'type': 'surface'}).tolist()

    row_height = 600


    total_height = number_of_rows * row_height
    combined_fig = make_subplots(rows=number_of_rows,
                                 cols=2,
                                 subplot_titles=titles,
                                 specs=specs,
                                 row_heights=[row_height]*number_of_rows)

    mins_fom = [df[fom_name].min() for df in dataframes]
    maxs_fom = [df[fom_name].max() for df in dataframes]
    global_min = min(mins_fom)
    global_max = max(maxs_fom)

    len_colorbar = (1/number_of_rows)*0.98


    for index, (df, title, suffix, heatmap, scatter) in enumerate(zip(dataframes,
                                                                      titles,
                                                                      suffixes,
                                                                      heatmaps_input,
                                                                      scatter_inputs)):



        row_index = index // 2 + 1
        col_index = index % 2 + 1

        if global_normalization:
            local_min = global_min
            local_max = global_max
        else:
            local_min = mins_fom[index]
            local_max = maxs_fom[index]

        fig = plot_interpolated_heatmap(df=df,
                                           x_name=x_name,
                                           y_name=y_name,
                                           fom_name=fom_name,
                                           bounds_x=bounds_x,
                                           bounds_y=bounds_y,
                                           heatmap_input=heatmap,
                                           scatter_input=scatter,
                                           add_trajectories=add_trajectories,
                                           names_suffix=suffix,
                                           title=title,
                                           min_value=local_min,
                                           max_value=local_max,
                                           grid_resolution=1000,
                                           minimization=minimization,
                                           in_3d=in_3d,
                                        colormap_name=colormap_name,
                                        skip_heatmap=skip_heatmaps,
                                        skip_scatter=skip_scatters)
        # print('col index:',col_index)
        # x_colorbrar = None
        # if not in_3d:
        if col_index == 1:
            x_colorbrar = 0.45
        else:
            x_colorbrar = 1.0

        #y_colorbar = 1 - (row_index - 0.5) / number_of_rows


        axis_index = 2*(row_index-1) + col_index
        y_colorbar = None
        #TODO FBM: figure this out better.
        try:
            if axis_index == 1:
                yaxis_name = 'yaxis'
            else:
                yaxis_name = f'yaxis{axis_index}'
            domain = combined_fig.layout[yaxis_name].domain  # e.g., (0.66, 0.88)
            if domain is None:
                y_colorbar = None
            else:
                y_colorbar = 0.5 * (domain[0] + domain[1])

        except(Exception) as err:
            #print(err)
            pass



        colorbar_dict = dict(x=x_colorbrar, y=y_colorbar, len=len_colorbar, yanchor='middle')
        for trace in fig.data:
            if "coloraxis" in trace:
                try:
                    trace.update(
                        dict(name=f'coloraxis_r-{row_index}_c-{col_index}',
                             colorbar=colorbar_dict,
                             colorscale=colormap_name,
                             zmin=local_min,
                             zmax=local_max))
                except(ValueError):
                    trace.update(
                        dict(name=f'coloraxis_r-{row_index}_c-{col_index}',
                             colorbar=colorbar_dict,
                             colorscale=colormap_name,
                             cmin=local_min,
                             cmax=local_max))
            combined_fig.add_trace(trace,
                                   row=row_index,
                                   col=col_index)
            #add xlabel and ylabel to the axis
            combined_fig.update_xaxes(title_text=_gnvn(x_name), row=row_index, col=col_index)
            combined_fig.update_yaxes(title_text=_gnvn(y_name), row=row_index, col=col_index)

    if len(dataframes) <= 6:
        legend = dict(
            orientation='h',
            x=0.5,  # Horizontal position (0 = far left, 1 = far right)
            y=1.15,  # Vertical position (0 = bottom, 1 = top)
            xanchor="center",  # Anchor point for the legend box
            yanchor="top",  # Vertical alignment
            bgcolor="rgba(255, 255, 255, 0.5)",  # Optional: Add a semi-transparent background
            bordercolor="black",  # Optional: Add a border color
            borderwidth=1  # Optional: Border thickness
        )
    else:
        legend = None

    combined_fig.update_layout(
        height=total_height,  # Adjust as needed
        template="plotly",
        legend=legend

    )
    return combined_fig






def get_grid_histogram(df:pd.DataFrame,
                         x_name:str,
                         y_name:str,
                         x_bins: int=None,
                         y_bins: int=None,
                         title: str = None,
                         suffix: str = None,
                         colormap_name='YlGnBu',
                         ):



    xs = df[x_name]
    ys = df[y_name]

    if x_bins is None:
        x_bins = xs//10
    if y_bins is None:
        y_bins = ys//10

    hovertemplate = (f"{_gnvn(x_name)}: %{{x}}"
                     f"<br>{_gnvn(y_name)}: %{{y}}"
                     f"<br>Number of Points: %{{z}}<extra></extra>")

    heatmap, x_edges, y_edges = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
    heatmap[heatmap == 0] = np.nan

    # Get the bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    heatmap = go.Heatmap(
        x=x_centers,
        y=y_centers,
        z=heatmap.T,  # Transpose for correct orientation
        colorscale=colormap_name,  # Choose a colorscale
        colorbar=dict(title="Number of Points"),
        hovertemplate=hovertemplate,
    )
    return heatmap

def plot_grid_histogram(df:pd.DataFrame,
                         x_name:str,
                         y_name:str,
                         x_bins: int=None,
                         y_bins: int=None,
                         title: str = None,
                         suffix: str = None,
                         colormap_name='YlGnBu',
                         ) -> go.Figure:
    heatmap = get_grid_histogram(df,
                                  x_name=x_name,
                                  y_name=y_name,
                                  x_bins=x_bins,
                                  y_bins=y_bins,
                                  title=title,
                                  suffix=suffix,
                                  colormap_name=colormap_name,
                                  )

    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title=title,
        xaxis_title=x_name,
        yaxis_title=y_name,
        template="plotly_white"
    )
    return fig

def plot_analytical_betas(df:pd.DataFrame,
                          x_name:str,
                          fom_name:str,
                          bounds_x:Tuple[float, float]=None,
                          title:str = None,
                          names_suffix: str = None)-> go.Figure:
    markersize_scatter = 2
    scatter_markers_dict = dict(
                    color='black',
                    size=markersize_scatter,
                    symbol='circle',
                    line=dict(color="rgba(255, 255, 255, 0.5)", width=1)
                )
    line_dict = dict(
            color='black',
            dash='solid'
            )
    data = go.Scatter(x=df[x_name],
                      y=df[fom_name],
                      mode='lines+markers',
                      line=line_dict,
                      marker=scatter_markers_dict,
                      zorder=1)
    fig = go.Figure(data=data)
    fig.update_layout(title=title,
                      xaxis_title=f"{_gnvn(x_name)}",
                      yaxis_title=f"{_gnvn(fom_name)}",
                      xaxis=dict(range=bounds_x),
                      template="plotly")
    return fig
    





