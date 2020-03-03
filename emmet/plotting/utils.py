from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.electronic_structure.core import Spin
from pymatgen.core.periodic_table import Element

import plotly.graph_objs as go
import plotly.subplots as tls

import scipy.interpolate as scint
import numpy as np
import copy

import networkx as nx


class BSPlotterPlotly:
    def __init__(self, bandstructure=None, dos=None, path_type=None):
        if path_type is not None:
            self._bs = self._format_bs(bandstructure, path_type)
        else:
            self._bs = bandstructure
        self._dos = dos

    @staticmethod
    def _format_bs(bandstructure, path_type):
        """
        Extract band structure for a specific convention from a band structure
        generated using the 'all' convention in HighSymmKpath.

        Returns:
            bs : New BandStructureSymmLine object for the convention 
            given by 'path_type'
        """
        bs = copy.deepcopy(bandstructure)

        lm = HighSymmKpath(bs.structure, path_type="lm", symprec=0.1, angle_tolerance=5)

        sc = HighSymmKpath(bs.structure, path_type="sc", symprec=0.1, angle_tolerance=5)

        hin = HighSymmKpath(bs.structure, path_type="hin", symprec=0.1, angle_tolerance=5)

        # -- Format branches to account for points close enough
        old_branches = bs.branches
        new_branches = []
        skip = False
        for branch_num in range(len(old_branches) - 1):
            if not skip:

                name0 = old_branches[branch_num]["name"].split("-")
                s_index0 = old_branches[branch_num]["start_index"]
                e_index0 = old_branches[branch_num]["end_index"]

                name1 = old_branches[branch_num + 1]["name"].split("-")
                s_index1 = old_branches[branch_num + 1]["start_index"]
                e_index1 = old_branches[branch_num + 1]["end_index"]

                if name0[0] == name0[1] and name1[0] == name1[1]:
                    if name0 != name1:
                        new_branches.append(
                            {"start_index": e_index0, "end_index": s_index1, "name": "{}-{}".format(name0[0], name1[0])}
                        )

                    else:
                        num_equal = 0
                        while old_branches[branch_num]["name"] == old_branches[branch_num + num_equal]["name"]:
                            num_equal += 1

                        name2 = old_branches[branch_num + 2]["name"].split("-")
                        e_index2 = old_branches[branch_num + 2]["end_index"]
                        new_branches.append(
                            {"start_index": e_index0, "end_index": e_index2, "name": "{}-{}".format(name0[0], name2[0])}
                        )

                        skip = True

                # if name0[0] == name0[1] and name1[0] == name1[1] and \
                #    s_index0 == e_index0 and s_index1 == e_index1:

                #     new_branches.append({'start_index': s_index0,
                #                          'end_index': e_index1,
                #                          'name': '{}-{}'.format(name0[0], name1[1])})
                #     skip = True

                elif name0[0] == name0[1] and name1[0] != name1[1]:
                    pass
                else:
                    new_branches.append(old_branches[branch_num])

                    if branch_num == len(old_branches) - 2:
                        new_branches.append(old_branches[branch_num + 1])

            else:
                skip = False

        bs.branches = new_branches

        # -- Determine cut of 'all' band structure to take
        path_lengths = []
        for bs_type in [lm, sc, hin]:

            total_points_path = 0
            total_seg_path = 0
            for seg in bs_type.kpath["path"]:
                total_points_path += len(seg)
                total_seg_path += 1

            path_lengths.append((total_seg_path, total_points_path))

        if path_type == "lm":
            path_length = path_lengths[0][1]
            path_length_before = 0

            num_breaks_before = 0
            num_breaks = path_lengths[0][0] - 1

            min_branch = 0
            max_branch = min_branch + path_length - 1 - num_breaks

            new_labels = [entry for block in lm.kpath["path"] for entry in block]

        elif path_type == "sc":
            path_length_before = path_lengths[0][1]
            path_length = path_lengths[1][1]

            num_breaks_before = path_lengths[0][0] - 1
            num_breaks = path_lengths[1][0] - 1

            min_branch = path_length_before - num_breaks_before - 1
            max_branch = min_branch + path_length - 1 - num_breaks

            new_labels = [entry for block in sc.kpath["path"] for entry in block]

        elif path_type == "hin":
            path_length_before = path_lengths[0][1] + path_lengths[1][1]
            path_length = path_lengths[2][1]

            num_breaks_before = path_lengths[0][0] + path_lengths[1][0] - 1
            num_breaks = path_lengths[2][0] - 1

            min_branch = path_length_before - num_breaks_before - 1
            max_branch = min_branch + path_length - 1 - num_breaks

            new_labels = [entry for block in hin.kpath["path"] for entry in block]

        bs.branches = bs.branches[min_branch:max_branch]

        # -- Shift branch indices according to cut
        shift_index = bs.branches[0]["start_index"]
        last_index = bs.branches[-1]["end_index"]

        for branch_num in range(len(bs.branches)):
            bs.branches[branch_num]["start_index"] += (-1) * shift_index
            bs.branches[branch_num]["end_index"] += (-1) * shift_index

        # -- Format band and distance data
        if bs.is_spin_polarized:
            new_bands = {
                Spin.up: np.zeros((bs.nb_bands, (last_index - shift_index + 1))),
                Spin.down: np.zeros((bs.nb_bands, (last_index - shift_index + 1))),
            }
        else:
            new_bands = {Spin.up: np.zeros((bs.nb_bands, (last_index - shift_index + 1)))}

        for band_num in range(bs.nb_bands):
            new_bands[Spin.up][band_num] = bs.bands[Spin.up][band_num][shift_index : (last_index + 1)]
            if bs.is_spin_polarized:
                new_bands[Spin.down][band_num] = bs.bands[Spin.down][band_num][shift_index : (last_index + 1)]

        bs.bands = new_bands
        bs.distance = [entry - bs.distance[shift_index] for entry in bs.distance[shift_index : (last_index + 1)]]

        # -- Format kpoints and insert appropriate labels
        bs.kpoints = bs.kpoints[shift_index : (last_index + 1)]

        label_inc = 0
        pre = "temp"
        for i, c in enumerate(bs.kpoints):
            if c.label is not None and c.label is not pre:

                if label_inc is not len(new_labels):
                    if i != (len(bs.kpoints) - 1) and bs.kpoints[i + 1].label == c.label:
                        bs.kpoints[i + 1]._label = new_labels[label_inc]

                    bs.kpoints[i]._label = new_labels[label_inc]

                    label_inc += 1
                pre = c.label

        return bs

    def _get_cont_path(self):
        """
        Refine kpath chosen to be completely continuous for plotting.

        Returns:
            distance_map : List containing the new mapping for distance segments
            in the refined path.
            kpath_euler: List of list of labels for new continous kpath
        """
        bs_data = self.bs_plot_data()

        # -- Strip latex math wrapping
        str_replace = {
            "$": "",
            "\\": "",
            "mid": "|",
            "Gamma": "\\Gamma",
            "GAMMA": "\\Gamma",
            "Sigma": "\\Sigma",
            "SIGMA": "\\Sigma",
        }

        # -- Get graph of kpoints
        G = nx.Graph()

        labels = []
        for point in self._bs.kpoints:
            if point.label is not None:
                label = point.label
                for key in str_replace.keys():
                    if key in point.label:
                        label = label.replace(key, str_replace[key])

                labels.append("$" + label + "$")

        plot_axis = []
        for i in range(int(len(labels) / 2)):
            G.add_edges_from([(labels[2 * i], labels[(2 * i) + 1])])
            plot_axis.append((labels[2 * i], labels[(2 * i) + 1]))

        # -- Eulerize graph and obtain Eulerian circuit
        G_euler = nx.algorithms.euler.eulerize(G)

        G_euler_circuit = nx.algorithms.euler.eulerian_circuit(G_euler)

        # -- Make a map for distance segments in band structure for plotting
        distances_map = []
        kpath_euler = []

        for edge_euler in G_euler_circuit:
            kpath_euler.append(edge_euler)
            for edge_reg in plot_axis:
                if edge_euler == edge_reg:
                    distances_map.append((plot_axis.index(edge_reg), False))
                elif edge_euler[::-1] == edge_reg:
                    distances_map.append((plot_axis.index(edge_reg), True))

        return distances_map, kpath_euler

    def _reg_traces(self, continuous=False, energy_window=(-7.0, 11.0)):
        """
        Obtain traces for the band structure data along with kpoint labels 
        and tick values for plotting.

        Returns:
            traces : List of plotly traces for the band structure data
            kpath_labels: List of labels for the kpoints
            tick_vals: List of values where ticks and labels should be
        """

        traces = []

        bs_data = self.bs_plot_data()

        # -- Strip latex math wrapping
        str_replace = {
            "$": "",
            "\\": "",
            "mid": "|",
            "Gamma": "\\Gamma",
            "GAMMA": "\\Gamma",
            "Sigma": "\\Sigma",
            "SIGMA": "\\Sigma",
        }

        for entry_num in range(len(bs_data["ticks"]["label"])):
            for key in str_replace.keys():
                if key in bs_data["ticks"]["label"][entry_num]:
                    bs_data["ticks"]["label"][entry_num] = bs_data["ticks"]["label"][entry_num].replace(
                        key, str_replace[key]
                    )
            bs_data["ticks"]["label"][entry_num] = "$" + bs_data["ticks"]["label"][entry_num] + "$"

        if continuous:
            distance_map, kpath_euler = self._get_cont_path()

            kpath_labels = [pair[0] for pair in kpath_euler]
            kpath_labels.append(kpath_euler[-1][1])

        else:
            distance_map = [(i, False) for i in range(len(bs_data["distances"]))]

            kpath_labels = []
            for label_ind in range(len(bs_data["ticks"]["label"]) - 1):
                if bs_data["ticks"]["label"][label_ind] != bs_data["ticks"]["label"][label_ind + 1]:
                    kpath_labels.append(bs_data["ticks"]["label"][label_ind])
            kpath_labels.append(bs_data["ticks"]["label"][-1])

        # -- Obtain bands to plot over
        bands = []
        for band_num in range(self._bs.nb_bands):
            if (bs_data["energy"][0][str(Spin.up)][band_num][0] <= energy_window[1]) and (
                bs_data["energy"][0][str(Spin.up)][band_num][0] >= energy_window[0]
            ):
                bands.append(band_num)

        pmin = 0.0
        tick_vals = [0.0]
        for (d, rev) in distance_map:

            x_dat = [dval - bs_data["distances"][d][0] + pmin for dval in bs_data["distances"][d]]

            pmin = x_dat[-1]

            for i in bands:

                if not rev:
                    y_dat = [bs_data["energy"][d][str(Spin.up)][i][j] for j in range(len(bs_data["distances"][d]))]
                elif rev:
                    y_dat = [
                        bs_data["energy"][d][str(Spin.up)][i][j] for j in reversed(range(len(bs_data["distances"][d])))
                    ]

                traces.append(
                    go.Scatter(
                        x=x_dat,
                        y=y_dat,
                        mode="lines",
                        line=dict(color=("#666666"), width=2),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                )

                if self._bs.is_spin_polarized:

                    if not rev:
                        y_dat = [
                            bs_data["energy"][d][str(Spin.down)][i][j] for j in range(len(bs_data["distances"][d]))
                        ]
                    else:
                        y_dat = [
                            bs_data["energy"][d][str(Spin.down)][i][j]
                            for j in reversed(range(len(bs_data["distances"][d])))
                        ]

                    traces.append(
                        go.Scatter(
                            x=x_dat,
                            y=y_dat,
                            mode="lines",
                            line=dict(color=("#666666"), width=2, dash="dash"),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                    )
            tick_vals.append(pmin)

        return traces, kpath_labels, tick_vals

    def _get_dos_data(self, orbital_projection=None, energy_window=(-7.0, 11.0)):
        """
        Obtain the total and projected density of states data and traces formatted 
            within a given energy window.
        Args:
            orbital_projection: String with selection of either total orbital projected data,
            or element specific data. Either None, "total", or Element.

        Returns:
            dosdata : List of plotly traces for the total and projected dos.
            pdos: Dictionary containing projected Dos objects
        """
        dosdata = []

        dos = self._dos

        dos_max = np.abs((dos.energies - dos.efermi - energy_window[1])).argmin()
        dos_min = np.abs((dos.energies - dos.efermi - energy_window[0])).argmin()

        if self._bs.is_spin_polarized:
            # Add second spin data if available
            trace_tdos = go.Scatter(
                x=dos.densities[Spin.down][dos_min:dos_max],
                y=dos.energies[dos_min:dos_max] - dos.efermi,
                mode="lines",
                name="Total DOS (spin ↓)",
                line=dict(color="#444444", dash="dash"),
                fill="tozerox",
            )

            dosdata.append(trace_tdos)

            tdos_label = "Total DOS (spin ↑)"
        else:
            tdos_label = "Total DOS"

        # total DOS
        trace_tdos = go.Scatter(
            x=dos.densities[Spin.up][dos_min:dos_max],
            y=dos.energies[dos_min:dos_max] - dos.efermi,
            mode="lines",
            name=tdos_label,
            line=dict(color="#444444"),
            fill="tozerox",
        )

        dosdata.append(trace_tdos)

        proj_dos = dos.get_element_dos()

        ele_dos = dos.get_element_dos()
        elements = [entry for entry in ele_dos.keys()]

        if orbital_projection == "total":
            proj_data = dos.get_spd_dos()
        elif type(orbital_projection) == Element:
            proj_data = dos.get_element_spd_dos(orbital_projection)
        elif orbital_projection is None:
            proj_data = ele_dos
        else:
            raise RuntimeError("Orbital projection selection is incorrect.")

        # Projected data
        count = 0
        colors = [
            "#1f77b4",  # muted blue
            "#ff7f0e",  # safety orange
            "#2ca02c",  # cooked asparagus green
            "#d62728",  # brick red
            "#9467bd",  # muted purple
            "#8c564b",  # chestnut brown
            "#e377c2",  # raspberry yogurt pink
            "#bcbd22",  # curry yellow-green
            "#17becf",  # blue-teal
        ]

        for label in proj_data.keys():

            if self._bs.is_spin_polarized:
                trace = go.Scatter(
                    x=proj_data[label].densities[Spin.down][dos_min:dos_max],
                    y=dos.energies[dos_min:dos_max] - dos.efermi,
                    mode="lines",
                    name=str(label) + " (spin ↓)",
                    line=dict(width=3, color=colors[count], dash="dash"),
                )

                dosdata.append(trace)
                spin_up_label = str(label) + " (spin ↑)"

            else:
                spin_up_label = str(label)

            trace = go.Scatter(
                x=proj_data[label].densities[Spin.up][dos_min:dos_max],
                y=dos.energies[dos_min:dos_max] - dos.efermi,
                mode="lines",
                name=spin_up_label,
                line=dict(width=3, color=colors[count]),
            )

            dosdata.append(trace)

            count += 1

        return dosdata, proj_data

    def bs_plot_data(self, zero_to_efermi=True):

        """
        Get the data nicely formatted for a plot

        Args:
            zero_to_efermi: Automatically subtract off the Fermi energy from the
                eigenvalues and plot.

        Returns:
            dict: A dictionary of the following format:
            ticks: A dict with the 'distances' at which there is a kpoint (the
            x axis) and the labels (None if no label).
            energy: A dict storing bands for spin up and spin down data
            [{Spin:[band_index][k_point_index]}] as a list (one element
            for each branch) of energy for each kpoint. The data is
            stored by branch to facilitate the plotting.
            vbm: A list of tuples (distance,energy) marking the vbms. The
            energies are shifted with respect to the fermi level is the
            option has been selected.
            cbm: A list of tuples (distance,energy) marking the cbms. The
            energies are shifted with respect to the fermi level is the
            option has been selected.
            lattice: The reciprocal lattice.
            zero_energy: This is the energy used as zero for the plot.
            band_gap:A string indicating the band gap and its nature (empty if
            it's a metal).
            is_metal: True if the band structure is metallic (i.e., there is at
            least one band crossing the fermi level).
        """
        distance = []
        energy = []
        if self._bs.is_metal():
            zero_energy = self._bs.efermi
        else:
            zero_energy = self._bs.get_vbm()["energy"]

        if not zero_to_efermi:
            zero_energy = 0.0

        for b in self._bs.branches:

            if self._bs.is_spin_polarized:
                energy.append({str(Spin.up): [], str(Spin.down): []})
            else:
                energy.append({str(Spin.up): []})
            distance.append([self._bs.distance[j] for j in range(b["start_index"], b["end_index"] + 1)])
            ticks = self.get_ticks()

            for i in range(self._bs.nb_bands):
                energy[-1][str(Spin.up)].append(
                    [self._bs.bands[Spin.up][i][j] - zero_energy for j in range(b["start_index"], b["end_index"] + 1)]
                )
            if self._bs.is_spin_polarized:
                for i in range(self._bs.nb_bands):
                    energy[-1][str(Spin.down)].append(
                        [
                            self._bs.bands[Spin.down][i][j] - zero_energy
                            for j in range(b["start_index"], b["end_index"] + 1)
                        ]
                    )

        vbm = self._bs.get_vbm()
        cbm = self._bs.get_cbm()

        vbm_plot = []
        cbm_plot = []

        for index in cbm["kpoint_index"]:
            cbm_plot.append(
                (self._bs.distance[index], cbm["energy"] - zero_energy if zero_to_efermi else cbm["energy"])
            )

        for index in vbm["kpoint_index"]:
            vbm_plot.append(
                (self._bs.distance[index], vbm["energy"] - zero_energy if zero_to_efermi else vbm["energy"])
            )

        bg = self._bs.get_band_gap()
        direct = "Indirect"
        if bg["direct"]:
            direct = "Direct"

        return {
            "ticks": ticks,
            "distances": distance,
            "energy": energy,
            "vbm": vbm_plot,
            "cbm": cbm_plot,
            "lattice": self._bs.lattice_rec.as_dict(),
            "zero_energy": zero_energy,
            "is_metal": self._bs.is_metal(),
            "band_gap": "{} {} bandgap = {}".format(direct, bg["transition"], bg["energy"])
            if not self._bs.is_metal()
            else "",
        }

    def get_ticks(self):
        """
        Get all ticks and labels for a band structure plot.

        Returns:
            dict: A dictionary with 'distance': a list of distance at which
            ticks should be set and 'label': a list of label for each of those
            ticks.
        """
        tick_distance = []
        tick_labels = []
        previous_label = self._bs.kpoints[0].label
        previous_branch = self._bs.branches[0]["name"]
        for i, c in enumerate(self._bs.kpoints):
            if c.label is not None:
                tick_distance.append(self._bs.distance[i])
                this_branch = None
                for b in self._bs.branches:
                    if b["start_index"] <= i <= b["end_index"]:
                        this_branch = b["name"]
                        break
                if c.label != previous_label and previous_branch != this_branch:
                    label1 = c.label
                    if label1.startswith("\\") or label1.find("_") != -1:
                        label1 = "$" + label1 + "$"
                    label0 = previous_label
                    if label0.startswith("\\") or label0.find("_") != -1:
                        label0 = "$" + label0 + "$"
                    tick_labels.pop()
                    tick_distance.pop()
                    tick_labels.append(label0 + "$\\mid$" + label1)
                else:
                    if c.label.startswith("\\") or c.label.find("_") != -1:
                        tick_labels.append("$" + c.label + "$")
                    else:
                        tick_labels.append(c.label)
                previous_label = c.label
                previous_branch = this_branch

        return {"distance": tick_distance, "label": tick_labels}

    def get_plotly_plot(self, smooth=False, smooth_tol=None, continuous=False, energy_window=(-7.0, 11.0)):
        """
        Get combined band structure and density of states plotly plot.

        Returns:
            disbandfig (Figure): A formatted plotly figure object.
        """
        dosbandfig = tls.make_subplots(rows=1, cols=2, shared_yaxes=True)

        # -- DOS Traces
        dostraces, dos = self._get_dos_data(energy_window=energy_window)

        # -- BS Traces
        btraces, kpath_labels, tick_vals = self._reg_traces(continuous, energy_window)

        # -- Add band traces
        for btrace in btraces:
            dosbandfig.append_trace(btrace, 1, 1)
        # -- Add dos traces
        for dostrace in dostraces:
            dosbandfig.append_trace(dostrace, 1, 2)

        xaxis_style = go.layout.XAxis(
            title=dict(text="$\\text{Wave Vector}$", font=dict(size=18)),
            tickmode="array",
            tickvals=tick_vals,
            ticktext=kpath_labels,
            tickfont=dict(size=18),
            ticks="inside",
            tickwidth=2,
            showgrid=True,
            gridcolor="rgb(192,192,192)",
            showline=True,
            linecolor="black",
            linewidth=2,
            mirror=True,
        )

        yaxis_style = go.layout.YAxis(
            title=dict(text="$\\text{E - E}_{\\text{f }} (\\text{eV})$", font=dict(size=18)),
            tickfont=dict(size=18),
            showgrid=False,
            zerolinecolor="black",
            showline=True,
            linecolor="black",
            zeroline=True,
            mirror="ticks",
            ticks="inside",
            linewidth=2,
            tickwidth=2,
            zerolinewidth=1.25,
            range=[-5, 9],
        )

        xaxis_style_dos = go.layout.XAxis(
            title=dict(text="$\\text{Density of States } (\\text{eV}^{-1})$", font=dict(size=18)),
            tickfont=dict(size=18),
            showgrid=False,
            showline=True,
            linecolor="black",
            # range=[0, 50],
            mirror=True,
            ticks="inside",
            linewidth=2,
            tickwidth=2,
        )

        yaxis_style_dos = go.layout.YAxis(
            tickfont=dict(size=18),
            showgrid=False,
            zerolinecolor="black",
            showline=True,
            linecolor="black",
            zeroline=True,
            mirror="ticks",
            ticks="inside",
            linewidth=2,
            tickwidth=2,
            zerolinewidth=1.25,
            range=[-5, 9],
        )

        layout = go.Layout(
            title="",
            xaxis1=xaxis_style,
            xaxis2=xaxis_style_dos,
            yaxis=yaxis_style,
            yaxis2=yaxis_style_dos,
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            width=1500,
            height=800,
        )

        dosbandfig["layout"].update(layout)

        legend = go.layout.Legend(
            x=1.01,
            y=1.00,
            font=dict(size=10),
            xanchor="left",
            yanchor="top",
            tracegroupgap=20,
            bordercolor="black",
            borderwidth=2,
        )

        dosbandfig["layout"]["legend"] = legend

        dosbandfig["layout"]["xaxis1"]["domain"] = [0.0, 0.6]
        dosbandfig["layout"]["xaxis2"]["domain"] = [0.65, 1.0]

        return dosbandfig
