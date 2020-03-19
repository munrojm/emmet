import io
import traceback
from datetime import datetime
from maggma.builders import Builder
from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.core import Spin, OrbitalType
from emmet.vasp.materials import structure_metadata as sm
import copy
import numpy as np

__author__ = "Jason Munro <jmunro@lbl.gov>"


class BSDOSBuilder(Builder):
    def __init__(self, tasks, materials, electronic_structure, bandstructures, dos, query=None, **kwargs):
        """
        Creates an electronic structure from a tasks collection, the associated band structures and density of states, and the materials structure

        This explicitly acts on bandstructure documents generated for the 'all' path type in HighSymmKpath.
        Individual bandstructures for each of the three conventions are generated.

        Really only usefull for MP Website infrastructure right now.

        materials (Store) : Store of materials documents
        electronic_structure  (Store) : Store of electronic structure documents
        bandstructures (Store) : store of bandstructures
        dos (Store) : store of DOS
        query (dict): dictionary to limit tasks to be analyzed
        """

        self.tasks = tasks
        self.materials = materials
        self.electronic_structure = electronic_structure
        self.bandstructures = bandstructures
        self.dos = dos
        self.query = query if query else {}

        super().__init__(sources=[tasks, materials, bandstructures, dos], targets=[electronic_structure], **kwargs)

        self.chunk_size = 1

    def get_items(self):
        """
        Gets all items to process

        Returns:
            generator or list relevant tasks and materials to process into materials documents
        """

        self.logger.info("Electronic Structure Builder Started")

        # get all materials without an electronic_structure document but bandstructure and dos fields
        # and there is either a dos or bandstructure
        q = dict(self.query)
        q["$and"] = [{"bandstructure.bs_task": {"$exists": 1}}, {"bandstructure.dos_task": {"$exists": 1}}]
        mat_ids = list(self.materials.distinct(self.materials.key, criteria=q))
        es_ids = self.electronic_structure.distinct("task_id")

        # get all materials that were updated since the electronic structure was last updated
        # and there is either a dos or bandstructure
        q = dict(self.query)
        q.update(
            {
                self.materials.last_updated_field: {
                    "$gt": self.materials._lu_func[1](self.electronic_structure.last_updated)
                }
            }
        )
        q["$and"] = [{"bandstructure.bs_task": {"$exists": 1}}, {"bandstructure.dos_task": {"$exists": 1}}]
        mats = [
            mat for mat in set(self.materials.distinct(self.materials.key, criteria=q)) | (set(mat_ids) - set(es_ids))
        ]

        chunk_size = 100
        mats_chunked = [mats[i : i + chunk_size] for i in range(0, len(mats), chunk_size)]

        self.logger.debug("Processing {} materials for electronic structure".format(len(mats)))

        self.total = len(mats_chunked)

        for chunk in mats_chunked:
            self.logger.debug("Handling materials: {}".format(chunk))
            mats = self._update_mat(chunk)
            yield mats

    def process_item(self, mats):
        """
        Process the band structures and dos data.

        Args:
            mat (dict): material document

        Returns:
            (dict): electronic_structure document
        """

        d_list = []

        for mat in mats:

            structure = Structure.from_dict(mat["structure"])

            structure_metadata = sm(structure)

            dos_data = {
                "total": {"band_gap": {}, "cbm": {}, "vbm": {}},
                "elements": {},
                "orbitals": {
                    "s": {"band_gap": {}, "cbm": {}, "vbm": {}},
                    "p": {"band_gap": {}, "cbm": {}, "vbm": {}},
                    "d": {"band_gap": {}, "cbm": {}, "vbm": {}},
                },
            }

            bs_data = {"total": {"band_gap": {}, "cbm": {}, "vbm": {}}}

            d = {
                self.electronic_structure.key: str(mat[self.materials.key]),
                "bandstructure": {
                    "sc": copy.deepcopy(bs_data),
                    "lm": copy.deepcopy(bs_data),
                    "hin": copy.deepcopy(bs_data),
                },
                "dos": dos_data,
            }

            self.logger.info("Processing: {}".format(mat[self.materials.key]))

            dos = CompleteDos.from_dict(mat["dos"]["data"])

            type_available = [bs_type for bs_type in mat["bandstructure"].keys() if mat["bandstructure"][bs_type]]
            self.logger.info("Processing band structure types: {}".format(type_available))

            for bs_type in type_available:

                bs = BandStructureSymmLine.from_dict(mat["bandstructure"][bs_type]["data"])
                is_sp = bs.is_spin_polarized

                # -- Get total and projected band structure data and traces
                d["bandstructure"][bs_type]["task_id"] = mat["bandstructure"][bs_type]["task_id"]
                d["bandstructure"][bs_type]["total"]["band_gap"] = bs.get_band_gap()
                d["bandstructure"][bs_type]["total"]["cbm"] = bs.get_cbm()
                d["bandstructure"][bs_type]["total"]["vbm"] = bs.get_vbm()

                self._process_bs(d=d, bs=bs, bs_type=bs_type)

                # -- Get total and projected dos data and traces
                if bs_type == "sc":
                    d["dos"]["task_id"] = mat["dos"]["task_id"]
                    self._process_dos(d=d, dos=dos, spin_polarized=is_sp)

            d = {
                self.electronic_structure.last_updated_field: mat[self.materials.last_updated_field],
                **structure_metadata,
                **d,
            }

            d_list.append(d)

        return d_list

    def update_targets(self, items):
        """
        Inserts the new task_types into the task_types collection

        Args:
            items ([([dict],[int])]): A list of tuples of materials to update and the corresponding processed task_ids
        """
        items = list(filter(None, items))[0]

        if len(items) > 0:
            self.logger.info("Updating {} electronic structure docs".format(len(items)))
            self.electronic_structure.update(items)
        else:
            self.logger.info("No electronic structure docs to update")

    @staticmethod
    def _process_bs(d, bs, bs_type):

        # -- Get equivalent labels between different conventions

        hskp = HighSymmKpath(bs.structure, path_type="all", symprec=0.1, angle_tolerance=5, atol=1e-5)
        eq_labels = hskp.equiv_labels

        if bs_type == "lm":
            gen_labels = set([label for label in eq_labels["lm"]["sc"]])
            kpath_labels = set([kpoint.label for kpoint in bs.kpoints if kpoint.label is not None])

            if not gen_labels.issubset(kpath_labels):
                new_structure = SpacegroupAnalyzer(bs.struct).get_primitive_standard_structure(
                    international_monoclinic=False
                )

                hskp = HighSymmKpath(new_structure, path_type="all", symprec=0.1, angle_tolerance=5, atol=1e-5)
                eq_labels = hskp.equiv_labels

        d["bandstructure"][bs_type]["total"]["equiv_labels"] = eq_labels[bs_type]

    @staticmethod
    def _process_dos(d, dos, spin_polarized):
        orbitals = [OrbitalType.s, OrbitalType.p, OrbitalType.d]

        ele_dos = dos.get_element_dos()
        tot_orb_dos = dos.get_spd_dos()

        for ele in ele_dos.keys():
            d["dos"]["elements"][str(ele)] = {}
            for sub_label in ["total", "s", "p", "d"]:
                d["dos"]["elements"][str(ele)][sub_label] = {"band_gap": {}, "cbm": {}, "vbm": {}}

        if spin_polarized:
            for s_ind, spin in enumerate([Spin.down, Spin.up]):
                # - Process total DOS data
                d["dos"]["total"]["band_gap"][spin] = dos.get_gap(spin=spin)
                (cbm, vbm) = dos.get_cbm_vbm(spin=spin)
                d["dos"]["total"]["cbm"][spin] = cbm
                d["dos"]["total"]["vbm"][spin] = vbm

                # - Process total orbital projection data
                for o_ind, orbital in enumerate(orbitals):
                    d["dos"]["orbitals"][str(orbital)]["band_gap"][spin] = tot_orb_dos[orbital].get_gap(spin=spin)
                    (cbm, vbm) = tot_orb_dos[orbital].get_cbm_vbm(spin=spin)
                    d["dos"]["orbitals"][str(orbital)]["cbm"][spin] = cbm
                    d["dos"]["orbitals"][str(orbital)]["vbm"][spin] = vbm

            # - Process element and element orbital projection data
            for ind1, ele in enumerate(ele_dos):
                orb_dos = dos.get_element_spd_dos(ele)

                for ind2, orbital in enumerate(["total"] + list(orb_dos.keys())):
                    if orbital == "total":
                        proj_dos = ele_dos
                        label = ele
                    else:
                        proj_dos = orb_dos
                        label = orbital

                    for spin in [Spin.down, Spin.up]:
                        d["dos"]["elements"][str(ele)][str(orbital)]["band_gap"][spin] = proj_dos[label].get_gap(
                            spin=spin
                        )
                        (cbm, vbm) = proj_dos[label].get_cbm_vbm(spin=spin)
                        d["dos"]["elements"][str(ele)][str(orbital)]["cbm"][spin] = cbm
                        d["dos"]["elements"][str(ele)][str(orbital)]["vbm"][spin] = vbm

        else:
            # - Process total DOS data
            d["dos"]["total"]["band_gap"][Spin.up] = dos.get_gap(spin=Spin.up)
            (cbm, vbm) = dos.get_cbm_vbm(spin=Spin.up)
            d["dos"]["total"]["cbm"][Spin.up] = cbm
            d["dos"]["total"]["vbm"][Spin.up] = vbm

            # - Process total orbital projection data
            for o_ind, orbital in enumerate(orbitals):
                d["dos"]["orbitals"][str(orbital)]["band_gap"][Spin.up] = tot_orb_dos[orbital].get_gap(spin=Spin.up)
                (cbm, vbm) = tot_orb_dos[orbital].get_cbm_vbm(spin=Spin.up)
                d["dos"]["orbitals"][str(orbital)]["cbm"][Spin.up] = cbm
                d["dos"]["orbitals"][str(orbital)]["vbm"][Spin.up] = vbm

            # - Process element and element orbital projection data
            for ind1, ele in enumerate(ele_dos):
                orb_dos = dos.get_element_spd_dos(ele)

                for ind2, orbital in enumerate(["total"] + orbitals):
                    if orbital == "total":
                        proj_dos = ele_dos
                        label = ele
                        ind = ind1 + 1
                    else:
                        proj_dos = orb_dos
                        label = orbital
                        ind = ind2

                    d["dos"]["elements"][str(ele)][str(orbital)]["band_gap"][Spin.up] = proj_dos[label].get_gap(
                        spin=Spin.up
                    )
                    (cbm, vbm) = proj_dos[label].get_cbm_vbm(spin=Spin.up)
                    d["dos"]["elements"][str(ele)][str(orbital)]["cbm"][Spin.up] = cbm
                    d["dos"]["elements"][str(ele)][str(orbital)]["vbm"][Spin.up] = vbm

    def _update_mat(self, mat_list):
        # find bs type for each task in task_type and store each different bs object

        mats = self.materials.query(
            properties=[self.materials.key, "structure", "inputs", "task_types", self.materials.last_updated_field],
            criteria={self.materials.key: {"$in": mat_list}},
        )

        mats_updated = []

        for mat in mats:

            mat["dos"] = {}
            mat["bandstructure"] = {"sc": {}, "lm": {}, "hin": {}}

            seen_bs_data = {"sc": [], "lm": [], "hin": []}
            seen_dos_data = []

            for task_id in mat["task_types"].keys():
                # obtain all bs calcs, their type, and last updated
                if "NSCF Line" in mat["task_types"][task_id]:

                    bs_type = None

                    task_query = self.tasks.query_one(
                        properties=["last_updated", "input.is_hubbard", "orig_inputs.kpoints"],
                        criteria={"task_id": int(task_id)},
                    )

                    bs = self.bandstructures.query_one(criteria={"metadata.task_id": int(task_id)})

                    structure = Structure.from_dict(bs["structure"])

                    bs_labels = bs["labels_dict"]

                    # - Find path type
                    if any([label.islower() for label in bs_labels]):
                        bs_type = "lm"
                    else:
                        for ptype in ["sc", "hin"]:
                            hskp = HighSymmKpath(
                                structure,
                                has_magmoms=False,
                                magmom_axis=None,
                                path_type=ptype,
                                symprec=0.1,
                                angle_tolerance=5,
                                atol=1e-5,
                            )
                            hs_labels_full = hskp.kpath["kpoints"]
                            hs_path_uniq = set([label for segment in hskp.kpath["path"] for label in segment])

                            hs_labels = {k: hs_labels_full[k] for k in hs_path_uniq if k in hs_path_uniq}

                            shared_items = {
                                k: bs_labels[k]
                                for k in bs_labels
                                if k in hs_labels and np.allclose(bs_labels[k], hs_labels[k], atol=1e-3)
                            }

                            if len(shared_items) == len(bs_labels) and len(shared_items) == len(hs_labels):
                                bs_type = ptype

                    is_hubbard = task_query["input"]["is_hubbard"]
                    nkpoints = task_query["orig_inputs"]["kpoints"]["nkpoints"]
                    lu_dt = task_query["last_updated"]

                    if bs_type is not None:
                        seen_bs_data[bs_type].append(
                            {
                                "task_id": str(task_id),
                                "is_hubbard": int(is_hubbard),
                                "nkpoints": int(nkpoints),
                                "updated_on": lu_dt,
                            }
                        )

                # Handle uniform tasks
                if "NSCF Uniform" in mat["task_types"][task_id]:
                    task_query = self.tasks.query_one(
                        properties=["last_updated", "input.is_hubbard", "orig_inputs.kpoints"],
                        criteria={"task_id": int(task_id)},
                    )

                    is_hubbard = task_query["input"]["is_hubbard"]
                    nkpoints = task_query["orig_inputs"]["kpoints"]["nkpoints"]
                    lu_dt = task_query["last_updated"]

                    seen_dos_data.append(
                        {
                            "task_id": str(task_id),
                            "is_hubbard": int(is_hubbard),
                            "nkpoints": int(nkpoints),
                            "updated_on": lu_dt,
                        }
                    )

            for bs_type in mat["bandstructure"]:
                # select "blessed" bs of each type
                if seen_bs_data[bs_type]:
                    sorted_data = sorted(
                        seen_bs_data[bs_type],
                        key=lambda entry: (entry["is_hubbard"], entry["nkpoints"], entry["updated_on"]),
                        reverse=True,
                    )

                    mat["bandstructure"][bs_type]["task_id"] = int(sorted_data[0]["task_id"])
                    mat["bandstructure"][bs_type]["data"] = self.bandstructures.query_one(
                        criteria={"metadata.task_id": int(sorted_data[0]["task_id"])}
                    )

            if seen_dos_data:

                sorted_dos_data = sorted(
                    seen_dos_data,
                    key=lambda entry: (entry["is_hubbard"], entry["nkpoints"], entry["updated_on"]),
                    reverse=True,
                )

                mat["dos"]["task_id"] = int(sorted_dos_data[0]["task_id"])
                mat["dos"]["data"] = self.dos.query_one(
                    criteria={"metadata.task_id": int(sorted_dos_data[0]["task_id"])}
                )

            mats_updated.append(mat)

        return mats_updated


class BSDOSPlotBuilder:
    pass
