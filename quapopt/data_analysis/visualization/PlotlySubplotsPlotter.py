# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Any, Union, Tuple, Dict, Iterator

import numpy as np
import pandas as pd
import plotly
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from quapopt.data_analysis.visualization import optimization_visualization as opt_vis
from quapopt import ancillary_functions as anf
from typing import Literal
from plotly.colors import qualitative

AxisType = Literal["x", "y", "z"]


@dataclass
class SubplotSpec:
    """Specification for a single subplot."""
    flat_idx: int
    grid_row: int
    grid_col: int
    filter_dict: Dict[str, Any]
    title: str
    color:str


class PlotlySubplotsPlotter:
    """
    A flexible subplot plotter that supports both single-variable and two-variable grid layouts.

    Parameters
    ----------
    results_dataframe : pd.DataFrame
        The dataframe containing the data to plot.
    subplot_variable_names : Union[str, Tuple[str, str]]
        Either a single variable name (str) for single-variable mode,
        or a tuple of (columns_variable, rows_variable) for two-variable mode.
    unique_values : Optional[Union[List[Any], Tuple[List[Any], List[Any]]]]
        For single-variable mode: List of values to include.
        For two-variable mode: Tuple of (column_values, row_values).
        If None, all unique values from the dataframe are used.
    n_cols : Optional[int]
        Number of columns in the grid.
        - Single-variable mode: defaults to 2, n_rows is computed automatically.
        - Two-variable mode: ignored (determined by unique column values).
    in_3d : bool
        Whether to create 3D surface subplots.

    Examples
    --------
    Single-variable mode (4 values displayed in 2x2 grid):
        plotter = PlotlySubplotsPlotter(df, subplot_variable_names="n_qubits", n_cols=2)

    Two-variable mode (columns=depth, rows=n_qubits):
        plotter = PlotlySubplotsPlotter(df, subplot_variable_names=("depth", "n_qubits"))
    """

    def __init__(
            self,
            results_dataframe: pd.DataFrame,
            subplot_variable_names: Optional[Union[str, Tuple[str], Tuple[str, str]]],
            unique_values: Optional[Union[List[Any], Tuple[List[Any], List[Any]]]] = None,
            n_cols: int = 2,
            in_3d: bool = False,
            root_figures_folder: Optional[str] = None
    ):
        df_res = results_dataframe.copy()

        if isinstance(subplot_variable_names, tuple):
            if len(subplot_variable_names) == 1:
                subplot_variable_names = subplot_variable_names[0]

        # Determine mode based on input type
        if isinstance(subplot_variable_names, str) or subplot_variable_names is None:
            self._is_single_variable_mode = True
            self._variable_names = (subplot_variable_names,)
        elif isinstance(subplot_variable_names, tuple) and len(subplot_variable_names) == 2:
            self._is_single_variable_mode = False
            self._variable_names = subplot_variable_names  # (columns_var, rows_var)
        else:
            raise ValueError(
                "subplot_variable_names must be either a string (single variable) "
                "or a tuple of two strings (columns_variable, rows_variable)."
            )


        # Initialize colors
        self._colors = dict(enumerate(plotly.colors.qualitative.Plotly))


        # Process unique values and filter dataframe
        if self._is_single_variable_mode:
            var_name = self._variable_names[0]

            if var_name is None:
                assert unique_values is None, "unique_values must be None when var_name is None"
                unique_vals = [None]

            else:
                if unique_values is None:
                    unique_vals = sorted(df_res[var_name].unique(), reverse=True)
                else:
                    unique_vals = list(unique_values)
                    df_res = df_res[df_res[var_name].isin(unique_vals)].copy()

            # Compute grid dimensions for single-variable mode
            n_values = len(unique_vals)

            self._n_cols = n_cols
            self._n_rows = math.ceil(n_values / self._n_cols)

            # Build subplot specs
            self._subplot_specs = []
            for flat_idx, val in enumerate(unique_vals):
                grid_row = flat_idx // self._n_cols
                grid_col = flat_idx % self._n_cols

                title_idx = f"{var_name}={val}" if var_name is not None else ""

                self._subplot_specs.append(SubplotSpec(
                    flat_idx=flat_idx,
                    grid_row=grid_row,
                    grid_col=grid_col,
                    filter_dict={var_name: val},
                    title=title_idx,
                    color=self.get_color(flat_idx)
                ))



        else:
            # Two-variable mode
            cols_var, rows_var = self._variable_names

            if unique_values is None:
                unique_col_vals = sorted(df_res[cols_var].unique(), reverse=True)
                unique_row_vals = sorted(df_res[rows_var].unique(), reverse=True)
            else:
                unique_col_vals, unique_row_vals = unique_values
                df_res = df_res[
                    df_res[cols_var].isin(unique_col_vals) &
                    df_res[rows_var].isin(unique_row_vals)
                    ].copy()

            self._n_cols = len(unique_col_vals)
            self._n_rows = len(unique_row_vals)

            # Build subplot specs (row-major order for plotly)
            self._subplot_specs = []
            flat_idx = 0
            for row_idx, row_val in enumerate(unique_row_vals):
                for col_idx, col_val in enumerate(unique_col_vals):

                    title_idx = ""

                    if cols_var is not None:
                        title_idx += f"{cols_var}={col_val},"
                    if rows_var is not None:
                        title_idx += f"{rows_var}={row_val}"



                    self._subplot_specs.append(SubplotSpec(
                        flat_idx=flat_idx,
                        grid_row=row_idx,
                        grid_col=col_idx,
                        filter_dict={cols_var: col_val, rows_var: row_val},
                        title=title_idx,
                        color=self.get_color(flat_idx)
                    ))
                    flat_idx += 1

        self.results_dataframe = df_res
        self._in_3d = in_3d

        # Initialize plotly
        plotly.io.templates.default = "plotly"
        plotly.offline.init_notebook_mode(connected=True)


        if root_figures_folder is None:
            root_figures_folder = anf.mirror_folder_path_in_root_output_folder()

        self._main_figures_folder = root_figures_folder

        #print('yo',self._main_figures_folder)

        os.makedirs(self._main_figures_folder, exist_ok=True)



    @property
    def n_subplots(self) -> int:
        """Total number of subplots."""
        return len(self._subplot_specs)

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Grid dimensions as (n_rows, n_cols)."""
        return self._n_rows, self._n_cols

    @property
    def is_single_variable_mode(self) -> bool:
        """Whether the plotter is in single-variable mode."""
        return self._is_single_variable_mode

    @property
    def in_3d(self) -> bool:
        return self._in_3d

    @property
    def main_figures_folder(self):
        return self._main_figures_folder

    def set_colors(self, colors_iterable: List[str]) -> None:
        """Set custom colors for the subplots."""
        self._colors = dict(enumerate(colors_iterable))

    def get_color(self, idx: int) -> str:
        """Get color for a given index (wraps around if needed)."""

        return self._colors[idx % len(self._colors)]

    def get_subplot_spec(self, flat_idx: int) -> Optional[SubplotSpec]:
        """Get subplot specification by flat index."""
        if 0 <= flat_idx < len(self._subplot_specs):
            return self._subplot_specs[flat_idx]
        return None

    def get_subplot_df(self, flat_idx: int) -> Optional[pd.DataFrame]:
        """
        Get filtered dataframe for a subplot by flat index.

        Parameters
        ----------
        flat_idx : int
            The flat index of the subplot (0 to n_subplots-1).

        Returns
        -------
        Optional[pd.DataFrame]
            Filtered dataframe for this subplot, or None if index is invalid.
        """
        spec = self.get_subplot_spec(flat_idx)
        if spec is None:
            return None

        mask = pd.Series(True, index=self.results_dataframe.index)
        for col, val in spec.filter_dict.items():
            if col is None or val is None:
                continue

            mask &= (self.results_dataframe[col] == val)

        return self.results_dataframe[mask]

    def get_subplot_df_by_grid(self, row_idx: int, col_idx: int) -> Optional[pd.DataFrame]:
        """
        Get filtered dataframe for a subplot by grid position.

        Parameters
        ----------
        row_idx : int
            Row index in the grid (0-indexed).
        col_idx : int
            Column index in the grid (0-indexed).

        Returns
        -------
        Optional[pd.DataFrame]
            Filtered dataframe for this subplot, or None if position is invalid.
        """
        flat_idx = row_idx * self._n_cols + col_idx
        return self.get_subplot_df(flat_idx)

    def iter_subplots(self) -> Iterator[Tuple[SubplotSpec, pd.DataFrame]]:
        """
        Iterate over all subplots.

        Yields
        ------
        Tuple[SubplotSpec, pd.DataFrame]
            The subplot specification and corresponding filtered dataframe.
        """
        for spec in self._subplot_specs:
            df = self.get_subplot_df(spec.flat_idx)
            yield spec, df

    def create_subplots_figure(self,
                               row_height: int = 1000,
                               column_width: int = 1000,
                               vertical_spacing: float = 0.1,
                               horizontal_spacing: float = 0.15,
                               figure_title: Optional[str] = None,
                               font_size: int = 32,
                               number_of_columns = None,
                               number_of_rows = None,
                               subplot_titles = None
                               ) -> Figure:
        """
        Create an empty plotly figure with the configured subplot grid.

        Returns
        -------
        Figure
            Plotly figure with subplot grid.
        """

        if number_of_rows is None:
            number_of_rows = self._n_rows
        if number_of_columns is None:
            number_of_columns = self._n_cols

        if self._in_3d:
            specs = np.full((number_of_rows, number_of_columns), dict(type='surface')).tolist()
        else:
            specs = None

        if subplot_titles is None:
            # Get titles in row-major order (as plotly expects)
            subplot_titles = [spec.title for spec in self._subplot_specs]

        # Pad with empty strings if grid has more cells than subplots
        total_cells = number_of_rows * number_of_columns
        if len(subplot_titles) < total_cells:
            subplot_titles.extend([''] * (total_cells - len(subplot_titles)))

        _fig_height = row_height * number_of_rows + 200
        _fig_width = column_width * number_of_columns

        figure = make_subplots(
            rows=number_of_rows,
            cols=number_of_columns,
            shared_yaxes=False,
            specs=specs,
            subplot_titles=subplot_titles,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing
        )

        figure.update_layout(height=_fig_height,
                             width=_fig_width,
                             title=figure_title,
                             title_x=0.5,
                             font=dict(size=font_size),
                             # leave some space at the top for the title
                             # the command is:
                             margin=dict(t=200)
                             )
        figure.update_annotations(font_size=font_size)

        return figure

    def update_heatmap_style(self,
                             figure: Figure,
                             heatmap_trace: plotly.graph_objs.Heatmap,
                             row_index: int,
                             column_index: int,
                             skip_hover_info: bool = True,
                             len_fraction: float = 0.85,
                             colormap_title: Optional[str] = None,
                             ):

        heatmap_trace = heatmap_trace.update(showlegend=False,
                                             # turn off hover info
                                             hoverinfo='skip' if skip_hover_info else 'all',
                                             colorbar=dict(
                                                 **opt_vis.get_colorbar_relative_position_in_subplots(figure,
                                                                                                      total_cols=self._n_cols,
                                                                                                      row=row_index,
                                                                                                      col=column_index,
                                                                                                      len_fraction=len_fraction,
                                                                                                      in_3d=self._in_3d),
                                                 title=colormap_title
                                             )
                                             )
        return heatmap_trace

    @staticmethod
    def get_plotly_row_col_from_spec(spec: SubplotSpec, ):
        return spec.grid_row + 1, spec.grid_col + 1





    def get_plotly_row_col_from_flat_index(self, flat_idx: int) -> Tuple[int, int]:
        """
        Get plotly row/col indices (1-indexed) for a flat index.

        Parameters
        ----------
        flat_idx : int
            The flat index of the subplot.

        Returns
        -------
        Tuple[int, int]
            (row, col) for use with plotly's add_trace (1-indexed).
        """
        spec = self.get_subplot_spec(flat_idx)
        if spec is None:
            raise IndexError(f"Invalid flat_idx: {flat_idx}")
        # Plotly uses 1-indexed row/col
        return self.get_plotly_row_col_from_spec(spec)

    @staticmethod
    def update_figure_with_standard_specs(figure:Figure,
                                          overwrite_kwargs_layout:Optional[Dict[str,Any]]=None,
                                          overwrite_kwargs_axes: Optional[Dict[str, Any]] = None,

                                          ):

        overwrite_kwargs_layout = overwrite_kwargs_layout or {}
        overwrite_kwargs_axes = overwrite_kwargs_axes or {}

        fontsize = int(32 * 2)
        font_family = 'Serif'


        default_kwargs_axis = dict(showgrid=True,
                                   zeroline=False,
                                   showline=True,
                                   mirror=True,
                                   tickwidth=2,
                                   tickfont=dict(size=fontsize, family=font_family),
                                   ticks="inside",
                                   tickson="boundaries",
                                   ticklen=10,
                                   linewidth=1,
                                   linecolor='black',
                                   gridcolor='rgba(128, 128, 128, 0.2)',
                                   )
        kwargs_axes = dict(default_kwargs_axis, **overwrite_kwargs_axes)


        figure.update_xaxes(**kwargs_axes)
        figure.update_yaxes(**kwargs_axes)


        default_kwargs_layout = dict(margin=dict(t=0, b=0, l=10, r=0),
                                     title_x=0.5,
                                     paper_bgcolor='rgba(255, 255, 255, 1.0)',
                                     plot_bgcolor='rgba(255, 255, 255, 1.0)',
                                     font=dict(size=fontsize, family=font_family),
                                     legend=dict(
                                         bgcolor='rgba(255, 255, 255, 0.5)',
                                         bordercolor='rgba(0, 0, 0, 0.5)',
                                         borderwidth=1,
                                         font=dict(size=fontsize, family=font_family),
                                         itemwidth=40,
                                         indentation=10,
                                         tracegroupgap=10,
                                         title_text=None,
                                     ),
                                     )

        all_kwargs = dict(default_kwargs_layout, **overwrite_kwargs_layout)


        figure.update_layout(**all_kwargs)

        figure.update_xaxes(title_standoff=30)
        figure.update_yaxes(title_standoff=45)

        return figure





        # return figure






    def _update_axis_2d(self,
                        figure: Figure,
                        which_axis: AxisType,
                        specs: SubplotSpec,
                        axis_range:Optional[Tuple[float,float]]=None,
                        axis_title: Optional[str] = None,
                        font_size=32,
                        axis_ticks:Optional[List[float]]=None,
                        ):

        row_index, col_index = self.get_plotly_row_col_from_spec(specs)

        kwargs = {'title_text': axis_title,
                  'row': row_index,
                  'col': col_index,
                  'tickfont': dict(size=font_size),
                  'range':axis_range,
                  #add ticks
                  'tickvals':axis_ticks
                  }



        if which_axis=='x':
            figure.update_xaxes(**kwargs)
        elif which_axis=='y':
            figure.update_yaxes(**kwargs)
        else:
            raise ValueError(f"Invalid axis type: {which_axis}")

        return figure

    def _update_axis_3d(self,
                        figure: Figure,
                        which_axis: AxisType,
                        specs: SubplotSpec,
                        axis_title: Optional[str] = None,
                        font_size=32,
                        axis_range:Optional[Tuple[float,float]]=None,
                        axis_ticks:Optional[List[float]]=None,
                        ):

        row_index, col_index = self.get_plotly_row_col_from_spec(specs)
        kwargs = {'title': axis_title,
                  'row': row_index,
                  'col': col_index,
                  'tickfont': dict(size=font_size),
                  'range':axis_range,
                  'tickvals':axis_ticks,
                  }

        flat_index = specs.flat_idx
        scene_key = "scene" if flat_index == 0 else f"scene{flat_index + 1}"
        axis_key = f'{which_axis}axis'

        figure.update_layout(**{scene_key: dict(**{axis_key: kwargs})})

        return figure

    def update_axis(self,
                    figure: Figure,
                    which_axis:AxisType,
                    specs: SubplotSpec,
                    axis_title: Optional[str] = None,
                    font_size=32,
                    axis_range:Optional[Tuple[float,float]]=None,
                    axis_ticks:Optional[List[float]]=None,
                    ):

        if which_axis == "z" and not self._in_3d:
            raise ValueError("z-axis only available in 3D mode")

        if self._in_3d:
            figure = self._update_axis_3d(figure=figure, which_axis=which_axis, specs=specs, axis_title=axis_title, font_size=font_size, axis_range=axis_range,
                                          axis_ticks=axis_ticks)
        else:
            figure = self._update_axis_2d(figure=figure, which_axis=which_axis, specs=specs, axis_title=axis_title, font_size=font_size, axis_range=axis_range,
                                          axis_ticks=axis_ticks)

        return figure

    @staticmethod
    def save_figure_static(
                    figure:Figure,
                    main_figures_folder:Optional[str]=None,
                    figure_subfolder:Optional[str]=None,
                    figure_filename:Optional[str]=None,
                    extension='png',
                    scale:int=2
                    ):

        figure_subfolder = figure_subfolder or ""
        figure_filename = figure_filename or f"Figure-{anf.create_random_uuid()}"
        main_figures_folder = main_figures_folder or anf.mirror_folder_path_in_root_output_folder()

        if figure_subfolder.startswith('/'):
            figure_subfolder = figure_subfolder[1:]


        figures_path = os.path.join(main_figures_folder, figure_subfolder)

        os.makedirs(figures_path, exist_ok=True)

        file_path_full = f"{figures_path}/{figure_filename}.{extension}"


        if extension in ['png', 'pdf']:
            plotly.io.write_image(figure, file_path_full, scale=scale)

        elif extension == 'html':
            plotly.offline.plot(figure,
                                filename=file_path_full)
        else:
            raise ValueError(f"Invalid extension: {extension}")




    def save_figure(self,
                    figure: Figure,
                    figure_subfolder: Optional[str] = None,
                    figure_filename: Optional[str] = None,
                    extension='png',
                    scale: int = 2
                    ):

        self.save_figure_static(figure=figure,
                                main_figures_folder=self._main_figures_folder,
                                figure_subfolder=figure_subfolder,
                                figure_filename=figure_filename,
                                extension=extension,scale=scale)