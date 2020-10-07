import os
import json
from bson.objectid import ObjectId
import msgpack
import zlib
from monty.msgpack import default as monty_default
from hashlib import md5
from maggma.builders import Builder
from pydash.objects import get
from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.electronic_structure.core import Spin, OrbitalType
from emmet.vasp.materials import structure_metadata as sm
import copy
import numpy as np

__author__ = "Jason Munro <jmunro@lbl.gov>"


class BSDOSBuilder(Builder):
    def __init__(
        self,
        tasks,
        materials,
        bandstructure,
        dos,
        bandstructure_fs,
        dos_fs,
        query=None,
        **kwargs,
    ):
        """
        Creates a bandstructure and dos collection from a tasks collection, 
        the associated band structures and density of states gridfs collections, 
        and the materials collection.

        Individual bandstructures for each of the three conventions are generated.

        materials (Store) : Store of materials documents
        bandstructure (Store) : Store of bandstructure summary data documents
        dos (Store) : Store of dos summary data documents
        bandstructure_fs (Store) : store of bandstructures
        dos_fs (Store) : store of DOS
        query (dict): dictionary to limit tasks to be analyzed
        """

        self.tasks = tasks
        self.materials = materials
        self.bandstructure = bandstructure
        self.dos = dos
        self.bandstructure_fs = bandstructure_fs
        self.dos_fs = dos_fs
        self.query = query if query else {}

        super().__init__(
            sources=[tasks, materials, bandstructure_fs, dos_fs],
            targets=[bandstructure, dos],
            **kwargs,
        )

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
        q["$and"] = [
            {"bandstructure.bs_task": {"$exists": 1}},
            {"bandstructure.dos_task": {"$exists": 1}},
        ]
        mat_ids = list(self.materials.distinct(self.materials.key, criteria=q))
        es_ids = self.bandstructure.distinct("task_id")

        # get all materials that were updated since the electronic structure was last updated
        # and there is either a dos or bandstructure
        q = dict(self.query)

        q["$and"] = [
            {"bandstructure.bs_task": {"$exists": 1}},
            {"bandstructure.dos_task": {"$exists": 1}},
        ]

        mats_set = set(
            self.bandstructure.newer_in(
                target=self.materials, criteria=q, exhaustive=True
            )
        ) | (set(mat_ids) - set(es_ids))

        mats = [mat for mat in mats_set]

        chunk_size = 10
        mats_chunked = [
            mats[i : i + chunk_size] for i in range(0, len(mats), chunk_size)
        ]

        self.logger.debug(
            "Processing {} materials for electronic structure".format(len(mats))
        )

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
                "elemental": {},
                "orbital": {
                    "s": {"band_gap": {}, "cbm": {}, "vbm": {}},
                    "p": {"band_gap": {}, "cbm": {}, "vbm": {}},
                    "d": {"band_gap": {}, "cbm": {}, "vbm": {}},
                },
            }

            bs_data = {"band_gap": {}, "cbm": {}, "vbm": {}, "nbands": {}}

            d = {
                "bandstructure": {
                    "sc": copy.deepcopy(bs_data),
                    "lm": copy.deepcopy(bs_data),
                    "hin": copy.deepcopy(bs_data),
                },
                "dos": dos_data,
            }

            self.logger.info("Processing: {}".format(mat[self.materials.key]))

            dos = CompleteDos.from_dict(mat["dos"]["data"])

            type_available = [
                bs_type
                for bs_type in mat["bandstructure"].keys()
                if mat["bandstructure"][bs_type]
            ]
            self.logger.info(
                "Processing band structure types: {}".format(type_available)
            )

            for index, bs_type in enumerate(type_available):

                bs = BandStructureSymmLine.from_dict(
                    mat["bandstructure"][bs_type]["data"]
                )
                is_sp = bs.is_spin_polarized

                # -- Get total and projected band structure data and traces
                d["bandstructure"][bs_type]["task_id"] = mat["bandstructure"][bs_type][
                    "task_id"
                ]
                d["bandstructure"][bs_type]["band_gap"] = bs.get_band_gap()
                d["bandstructure"][bs_type]["cbm"] = bs.get_cbm()
                d["bandstructure"][bs_type]["vbm"] = bs.get_vbm()
                d["bandstructure"][bs_type]["nbands"] = bs.nb_bands

                self._process_bs(
                    d=d, bs=bs, bs_type=bs_type, skip="lm" not in type_available
                )

                # -- Get total and projected dos data and traces
                if index == 0:
                    d["dos"]["total"]["task_id"] = mat["dos"]["task_id"]
                    self._process_dos(d=d, dos=dos, spin_polarized=is_sp)

            bs_data = {
                self.bandstructure.key: mat[self.materials.key],
                self.bandstructure.last_updated_field: mat[
                    self.materials.last_updated_field
                ],
                **structure_metadata,
                **d["bandstructure"],
            }

            dos_data = {
                self.dos.key: mat[self.materials.key],
                self.dos.last_updated_field: mat[self.materials.last_updated_field],
                **structure_metadata,
                **d["dos"],
            }

            d_list.append((bs_data, dos_data))

        return d_list

    def update_targets(self, items):
        """
        Inserts the new task_types into the task_types collection

        Args:
            items ([([dict],[int])]): A list of tuples of docs to update
        """
        items = list(filter(None, items))[0]

        if len(items) > 0:
            self.logger.info("Updating {} electronic structure docs".format(len(items)))

            for item in items:
                self.bandstructure.update(item[0])
                self.dos.update(item[1])
        else:
            self.logger.info("No electronic structure docs to update")

    def _process_bs(self, d, bs, bs_type, skip):

        if not skip:
            # -- Get equivalent labels between different conventions
            hskp = HighSymmKpath(
                bs.structure, path_type="all", symprec=0.1, angle_tolerance=5, atol=1e-5
            )
            eq_labels = hskp.equiv_labels

            if bs_type == "lm":
                gen_labels = set([label for label in eq_labels["lm"]["sc"]])
                kpath_labels = set(
                    [kpoint.label for kpoint in bs.kpoints if kpoint.label is not None]
                )

                if not gen_labels.issubset(kpath_labels):
                    new_structure = SpacegroupAnalyzer(
                        bs.structure
                    ).get_primitive_standard_structure(international_monoclinic=False)

                    hskp = HighSymmKpath(
                        new_structure,
                        path_type="all",
                        symprec=0.1,
                        angle_tolerance=5,
                        atol=1e-5,
                    )
                    eq_labels = hskp.equiv_labels

            d["bandstructure"][bs_type]["equiv_labels"] = eq_labels[bs_type]
        else:
            d["bandstructure"][bs_type]["equiv_labels"] = {}

    @staticmethod
    def _process_dos(d, dos, spin_polarized):
        orbitals = [OrbitalType.s, OrbitalType.p, OrbitalType.d]

        ele_dos = dos.get_element_dos()
        tot_orb_dos = dos.get_spd_dos()

        for ele in ele_dos.keys():
            d["dos"]["elemental"][str(ele)] = {}
            for sub_label in ["total", "s", "p", "d", "f"]:
                d["dos"]["elemental"][str(ele)][sub_label] = {
                    "band_gap": {},
                    "cbm": {},
                    "vbm": {},
                }

        if spin_polarized:
            for s_ind, spin in enumerate([Spin.down, Spin.up]):
                # - Process total DOS data
                d["dos"]["total"]["band_gap"][spin] = dos.get_gap(spin=spin)
                (cbm, vbm) = dos.get_cbm_vbm(spin=spin)
                d["dos"]["total"]["cbm"][spin] = cbm
                d["dos"]["total"]["vbm"][spin] = vbm

                # - Process total orbital projection data
                for o_ind, orbital in enumerate(orbitals):
                    d["dos"]["orbital"][str(orbital)]["band_gap"][spin] = tot_orb_dos[
                        orbital
                    ].get_gap(spin=spin)
                    (cbm, vbm) = tot_orb_dos[orbital].get_cbm_vbm(spin=spin)
                    d["dos"]["orbital"][str(orbital)]["cbm"][spin] = cbm
                    d["dos"]["orbital"][str(orbital)]["vbm"][spin] = vbm

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
                        d["dos"]["elemental"][str(ele)][str(orbital)]["band_gap"][
                            spin
                        ] = proj_dos[label].get_gap(spin=spin)
                        (cbm, vbm) = proj_dos[label].get_cbm_vbm(spin=spin)
                        d["dos"]["elemental"][str(ele)][str(orbital)]["cbm"][spin] = cbm
                        d["dos"]["elemental"][str(ele)][str(orbital)]["vbm"][spin] = vbm

        else:
            # - Process total DOS data
            d["dos"]["total"]["band_gap"][Spin.up] = dos.get_gap(spin=Spin.up)
            (cbm, vbm) = dos.get_cbm_vbm(spin=Spin.up)
            d["dos"]["total"]["cbm"][Spin.up] = cbm
            d["dos"]["total"]["vbm"][Spin.up] = vbm

            # - Process total orbital projection data
            for o_ind, orbital in enumerate(orbitals):
                d["dos"]["orbital"][str(orbital)]["band_gap"][Spin.up] = tot_orb_dos[
                    orbital
                ].get_gap(spin=Spin.up)
                (cbm, vbm) = tot_orb_dos[orbital].get_cbm_vbm(spin=Spin.up)
                d["dos"]["orbital"][str(orbital)]["cbm"][Spin.up] = cbm
                d["dos"]["orbital"][str(orbital)]["vbm"][Spin.up] = vbm

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

                    d["dos"]["elemental"][str(ele)][str(orbital)]["band_gap"][
                        Spin.up
                    ] = proj_dos[label].get_gap(spin=Spin.up)
                    (cbm, vbm) = proj_dos[label].get_cbm_vbm(spin=Spin.up)
                    d["dos"]["elemental"][str(ele)][str(orbital)]["cbm"][Spin.up] = cbm
                    d["dos"]["elemental"][str(ele)][str(orbital)]["vbm"][Spin.up] = vbm

    def _update_mat(self, mat_list):
        # find bs type for each task in task_type and store each different bs object

        mats = self.materials.query(
            properties=[
                self.materials.key,
                "structure",
                "inputs",
                "task_types",
                self.materials.last_updated_field,
            ],
            criteria={self.materials.key: {"$in": mat_list}},
        )

        mats_updated = []

        problem_tasks = []

        for mat in mats:

            mat["dos"] = {}
            mat["bandstructure"] = {"sc": {}, "lm": {}, "hin": {}}

            seen_bs_data = {"sc": [], "lm": [], "hin": []}
            seen_dos_data = []

            for task_id in mat["task_types"].keys():
                # obtain all bs calcs, their type, and last updated
                try:
                    if "NSCF Line" in mat["task_types"][task_id]:

                        bs_type = None

                        task_query = self.tasks.query_one(
                            properties=[
                                "last_updated",
                                "input.is_hubbard",
                                "orig_inputs.kpoints",
                            ],
                            criteria={"task_id": str(task_id)},
                        )

                        bs = self.bandstructure_fs.query_one(
                            criteria={"metadata.task_id": str(task_id)}
                        )

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
                                hs_path_uniq = set(
                                    [
                                        label
                                        for segment in hskp.kpath["path"]
                                        for label in segment
                                    ]
                                )

                                hs_labels = {
                                    k: hs_labels_full[k]
                                    for k in hs_path_uniq
                                    if k in hs_path_uniq
                                }

                                shared_items = {
                                    k: bs_labels[k]
                                    for k in bs_labels
                                    if k in hs_labels
                                    and np.allclose(
                                        bs_labels[k], hs_labels[k], atol=1e-3
                                    )
                                }

                                if len(shared_items) == len(bs_labels) and len(
                                    shared_items
                                ) == len(hs_labels):
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
                except Exception:
                    problem_tasks.append(task_id)
                    pass

                # Handle uniform tasks
                if "NSCF Uniform" in mat["task_types"][task_id]:
                    task_query = self.tasks.query_one(
                        properties=[
                            "last_updated",
                            "input.is_hubbard",
                            "orig_inputs.kpoints",
                        ],
                        criteria={"task_id": str(task_id)},
                    )

                    is_hubbard = task_query["input"]["is_hubbard"]

                    if (
                        task_query["orig_inputs"]["kpoints"]["generation_style"]
                        == "Monkhorst"
                    ):
                        nkpoints = np.prod(
                            task_query["orig_inputs"]["kpoints"]["kpoints"][0], axis=0
                        )
                    else:
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
                        key=lambda entry: (
                            entry["is_hubbard"],
                            entry["nkpoints"],
                            entry["updated_on"],
                        ),
                        reverse=True,
                    )

                    mat["bandstructure"][bs_type]["task_id"] = str(
                        sorted_data[0]["task_id"]
                    )
                    mat["bandstructure"][bs_type][
                        "object"
                    ] = self.bandstructure_fs.query_one(
                        criteria={"metadata.task_id": str(sorted_data[0]["task_id"])}
                    )

            if seen_dos_data:

                sorted_dos_data = sorted(
                    seen_dos_data,
                    key=lambda entry: (
                        entry["is_hubbard"],
                        entry["nkpoints"],
                        entry["updated_on"],
                    ),
                    reverse=True,
                )

                mat["dos"]["task_id"] = str(sorted_dos_data[0]["task_id"])
                mat["dos"]["object"] = self.dos_fs.query_one(
                    criteria={"metadata.task_id": str(sorted_dos_data[0]["task_id"])}
                )

            mats_updated.append(mat)

        return mats_updated


class DOSCopyBuilder(Builder):
    def __init__(self, dos, s3, query=None, **kwargs):
        """
        Copies dos objects from mongodb to an aws or minio s3 bucket

        dos (Store) : GridFS Store of dos data
        s3  (Store) : s3 store
        query (dict): dictionary to limit dos data to be analyzed
        """

        self.s3 = s3
        self.dos = dos
        self.query = query if query else {}
        self.s3.write_to_s3_in_process_items = True

        super().__init__(sources=[dos], targets=[s3], **kwargs)

        self.chunk_size = 5

    def get_items(self):
        """
        Gets all items to process

        Returns:
            list relevant  dos objects
        """

        self.logger.info("DOS Copy Builder Started")

        q = dict(self.query)

        # All task_ids

        dos_oids = self.dos.distinct(field="_id", criteria=q)
        s3_oids = [
            ObjectId(entry) for entry in self.s3.distinct(field="gridfs_id", criteria=q)
        ]

        dos_set = set(dos_oids) - set(s3_oids)

        dos_list = [key for key in dos_set]

        self.logger.debug("Processing {} DOS objects for copying".format(len(dos_list)))

        self.total = len(dos_list)

        for entry in dos_list:
            metadata = self.dos._files_store.query_one(
                criteria={"_id": entry}, properties=["metadata", "uploadDate"],
            )
            data = {
                "gridfs_id": str(metadata["_id"]),
                "task_id": str(metadata["metadata"]["task_id"]),
                self.s3.last_updated_field: metadata["uploadDate"],
                "dos": self.dos.query_one(criteria={"_id": entry}),
            }

            yield data

    def process_item(self, entry):

        dos = CompleteDos.from_dict(entry["dos"])
        efermi = dos.efermi

        min_energy = min([min(dos.energies)])
        max_energy = max([max(dos.energies)])

        num_uniq_elements = len(set([site.species_string for site in dos.structure]))

        structure = Structure.from_dict(entry["dos"]["structure"])

        d = {
            "gridfs_id": entry["gridfs_id"],
            "object": entry["dos"],
            "task_id": entry["task_id"],
            "efermi": efermi,
            "min_energy": min_energy,
            "max_energy": max_energy,
            "num_uniq_elements": num_uniq_elements,
            self.s3.last_updated_field: entry["last_updated"],
        }

        return d

    def update_targets(self, items):
        """
        Copy each dos to the bucket

        Args:
            items ([([dict],[int])]): A list of tuples of materials to update and the corresponding processed task_ids
        """
        items = list(filter(None, items))

        if len(items) > 0:
            self.logger.info("Uploading {} band structures".format(len(items)))
            for item in items:
                self.s3.update(
                    [item], key=[key for key in item if key != "object"],
                )
        else:
            self.logger.info("No dos entries to copy")


class BSCopyBuilder(Builder):
    def __init__(
        self, bandstructures, tasks, s3, task_types=None, query=None, **kwargs
    ):
        """
        Copies band structure objects from mongodb to an aws or minio s3 bucket

        s3 (Store): s3 store
        bandstructures (Store) : store of bandstructures
        tasks (Store): store of tasks
        task_type (Store): Optional store of task type descriptions
        query (dict): dictionary to limit bandstructures to be analyzed
        """

        self.s3 = s3
        self.bandstructures = bandstructures
        self.tasks = tasks
        self.task_types = task_types
        self.query = query if query else {}

        super().__init__(
            sources=[bandstructures, tasks, task_types], targets=[s3], **kwargs
        )

        self.chunk_size = 5

    def get_items(self):
        """
        Gets all items to process

        Returns:
            list relevant band structures
        """

        self.logger.info("Band Structure Copy Builder Started")

        q = dict(self.query)

        # All task_ids

        bs_oids = self.bandstructures.distinct(field="_id", criteria=q)
        s3_oids = [
            ObjectId(entry) for entry in self.s3.distinct(field="gridfs_id", criteria=q)
        ]

        bs_set = set(bs_oids) - set(s3_oids)

        bs_list = [key for key in bs_set]

        self.logger.debug(
            "Processing {} band structure objects for copying".format(len(bs_list))
        )

        self.total = len(bs_list)

        for entry in bs_list:
            metadata = self.bandstructures._files_store.query_one(
                criteria={"_id": entry}, properties=["metadata", "uploadDate"],
            )

            task_data = self.tasks.query_one(
                {"task_id": metadata["metadata"]["task_id"]},
                ["task_type", "orig_inputs"],
            )

            # This handles old task documents without a task type description
            if task_data.get("task_type", None) is None:
                if self.task_types is None:
                    raise RuntimeError(
                        "Some task documents are missing task descriptions.\
                        Please address this or include a task_type store."
                    )
                temp_type = self.task_types.query_one(
                    {"task_id": metadata["metadata"]["task_id"]}, ["task_type"]
                ).get("task_type", None)

                task_data["task_type"] = temp_type

            bs = self.bandstructures.query_one(criteria={"_id": entry})

            if not bs.get("structure", None):
                bs["structure"] = task_data["orig_inputs"]["poscar"]["structure"]

            if task_data["task_type"] is not None:

                if "line" in task_data["task_type"].lower():
                    mode = "line"
                    if bs["labels_dict"] == {}:
                        labels = get(task_data, "orig_inputs.kpoints.labels", None)
                        kpts = get(task_data, "orig_inputs.kpoints.kpoints", None)
                        if labels and kpts:
                            labels_dict = dict(zip(labels, kpts))
                            labels_dict.pop(None, None)
                        else:
                            struc = Structure.from_dict(entry["bs"]["structure"])
                            labels_dict = HighSymmKpath(struc)._kpath["kpoints"]

                        bs["labels_dict"] = labels_dict
                else:
                    mode = "uniform"

                data = {
                    "gridfs_id": str(metadata["_id"]),
                    "task_id": str(metadata["metadata"]["task_id"]),
                    self.s3.last_updated_field: metadata["uploadDate"],
                    "bs": bs,
                    "mode": mode,
                }

                yield data

            else:
                pass

    def process_item(self, entry):

        mode = entry["mode"]

        bs = BandStructureSymmLine.from_dict(entry["bs"])

        spin_polarized = bs.is_spin_polarized
        num_bands = bs.nb_bands
        efermi = bs.efermi

        keys = list(bs.bands.keys())
        try:
            min_energy = min(
                [
                    min(np.array(bs.bands[keys[0]].item()["data"]).flatten())
                    for key in keys
                ]
            )
            max_energy = max(
                [
                    max(np.array(bs.bands[keys[0]].item()["data"]).flatten())
                    for key in keys
                ]
            )
        except Exception:
            min_energy = min([min(np.ndarray.flatten(bs.bands[key])) for key in keys])
            max_energy = max([max(np.ndarray.flatten(bs.bands[key])) for key in keys])

        structure = Structure.from_dict(entry["bs"]["structure"])

        num_uniq_elements = len(set([site.species_string for site in structure]))

        if mode == "line":

            bs_labels = entry["bs"]["labels_dict"]

            bs_type = "unknown"

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
                    hs_path_uniq = set(
                        [label for segment in hskp.kpath["path"] for label in segment]
                    )

                    hs_labels = {
                        k: hs_labels_full[k] for k in hs_path_uniq if k in hs_path_uniq
                    }

                    shared_items = {
                        k: bs_labels[k]
                        for k in bs_labels
                        if k in hs_labels
                        and np.allclose(bs_labels[k], hs_labels[k], atol=1e-3)
                    }

                    if len(shared_items) == len(bs_labels) and len(shared_items) == len(
                        hs_labels
                    ):
                        bs_type = ptype

            d = {
                "gridfs_id": entry["gridfs_id"],
                "object": entry["bs"],
                "mode": mode,
                "task_id": entry["task_id"],
                "spin_polarized": spin_polarized,
                "num_bands": num_bands,
                "path_type": bs_type,
                "efermi": efermi,
                "min_energy": min_energy,
                "max_energy": max_energy,
                "num_uniq_elements": num_uniq_elements,
                self.s3.last_updated_field: entry["last_updated"],
            }

        else:
            d = {
                "gridfs_id": entry["gridfs_id"],
                "object": entry["bs"],
                "mode": mode,
                "task_id": entry["task_id"],
                "spin_polarized": spin_polarized,
                "num_bands": num_bands,
                "efermi": efermi,
                "min_energy": min_energy,
                "max_energy": max_energy,
                "num_uniq_elements": num_uniq_elements,
                self.s3.last_updated_field: entry["last_updated"],
            }

        return d

    def update_targets(self, items):
        """
        Copy each band structure to the bucket

        Args:
            items ([([dict],[int])]): A list of tuples of materials to update and the corresponding processed task_ids
        """
        items = list(filter(None, items))

        if len(items) > 0:
            self.logger.info("Uploading {} band structures".format(len(items)))
            for item in items:
                self.s3.update([item], key=[key for key in item if key != "object"])
        else:
            self.logger.info("No band structure to copy")


class BSFileCopyBuilder(Builder):  # THIS IS FOR COPYING TO MINIO ONLY. NEEDS WORK.
    def __init__(
        self, tasks, bandstructures, s3, meta_dir, target_dir, query=None, **kwargs
    ):
        """
        Copies band structure and dos objects from mongodb to a minio bucket via direct direct file copy
        
        meta_dir (str): Directory for minio metadata
        target_dir (str): Directory for minio data
        bandstructures : Store of band structure documents
        s3  (Store) : S3 store defining minio index
        query (dict): dictionary to limit bandstructures to be analyzed
        """
        self.s3 = s3
        self.tasks = tasks
        self.bandstructures = bandstructures
        self.meta_dir = meta_dir
        self.target_dir = target_dir
        self.query = query if query else {}
        self.s3.write_to_s3_in_process_items = True

        super().__init__(
            sources=[self.tasks, self.bandstructures], targets=[self.s3], **kwargs
        )

        self.chunk_size = 5

    def get_items(self):
        """
        Gets all items to process

        Returns:
            list relevant bandstructure and dos objects
        """

        self.logger.info("Electronic Structure Copy Builder Started")

        q = dict(self.query)

        # All task_ids

        task_oids = self.tasks.distinct(
            field="calcs_reversed.0.bandstructure_fs_id",
            criteria={"task_label": "nscf uniform"},
        )  # only line-modes for now
        bs_oids = self.bandstructures.distinct(field="_id", criteria=q)
        s3_oids = [
            ObjectId(entry) for entry in self.s3.distinct(field="gridfs_id", criteria=q)
        ]

        bs_set = set(bs_oids) - set(task_oids) - set(s3_oids)

        bs_list = [key for key in bs_set]

        self.logger.debug(
            "Processing {} bandstructure objects for copying".format(len(bs_list))
        )

        self.total = len(bs_list)

        for entry in bs_list:
            metadata = self.bandstructures._files_store.query_one(
                criteria={"_id": entry}, properties=["metadata", "uploadDate"],
            )
            data = {
                "gridfs_id": str(metadata["_id"]),
                "task_id": str(metadata["metadata"]["task_id"]),
                self.s3.last_updated_field: metadata["uploadDate"],
                "bs": self.bandstructures.query_one(criteria={"_id": entry}),
            }

            yield data

    def process_item(self, entry):

        bs = BandStructureSymmLine.from_dict(entry["bs"])
        spin_polarized = bs.is_spin_polarized
        num_bands = bs.nb_bands
        efermi = bs.efermi

        keys = bs.bands.keys()

        min_energy = min([min(np.ndarray.flatten(bs.bands[key])) for key in keys])
        max_energy = max([max(np.ndarray.flatten(bs.bands[key])) for key in keys])

        num_uniq_elements = len(set([site.species_string for site in bs.structure]))

        structure = Structure.from_dict(entry["bs"]["structure"])

        bs_labels = entry["bs"]["labels_dict"]

        bs_type = "unknown"

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
                hs_path_uniq = set(
                    [label for segment in hskp.kpath["path"] for label in segment]
                )

                hs_labels = {
                    k: hs_labels_full[k] for k in hs_path_uniq if k in hs_path_uniq
                }

                shared_items = {
                    k: bs_labels[k]
                    for k in bs_labels
                    if k in hs_labels
                    and np.allclose(bs_labels[k], hs_labels[k], atol=1e-3)
                }

                if len(shared_items) == len(bs_labels) and len(shared_items) == len(
                    hs_labels
                ):
                    bs_type = ptype

        search_doc = {
            "gridfs_id": entry["gridfs_id"],
            "mode": "line",
            "task_id": entry["task_id"],
            "spin_polarized": str(spin_polarized),
            "num_bands": str(num_bands),
            "path_type": bs_type,
            "efermi": str(efermi),
            "min_energy": str(min_energy),
            "max_energy": str(max_energy),
            "num_uniq_elements": str(num_uniq_elements),
            self.s3.last_updated_field: entry["last_updated"],
        }

        data = {
            "gridfs_id": entry["gridfs_id"],
            "data": entry["bs"],
            "mode": "line",
            "task_id": entry["task_id"],
            "path_type": bs_type,
        }

        if self.target_dir is not None:
            data = msgpack.packb(data, default=monty_default)
            if self.s3.compress:
                search_doc["compression"] = "zlib"
                data = zlib.compress(data)

            meta_json = {
                "version": "1.0.2",
                "checksum": {"algorithm": "", "blocksize": 0, "hashes": None},
                "meta": {
                    "content-type": "application/octet-stream",
                    "etag": calc_etag(data),
                },
            }

            if not os.path.exists(
                f"{self.meta_dir}/{self.s3.sub_dir}/{search_doc['gridfs_id']}"
            ):
                os.makedirs(
                    f"{self.meta_dir}/{self.s3.sub_dir}/{search_doc['gridfs_id']}"
                )
            with open(
                f"{self.meta_dir}/{self.s3.sub_dir}/{search_doc['gridfs_id']}/fs.json",
                "w",
            ) as outfile:
                json.dump(meta_json, outfile)

            if not os.path.exists(f"{self.target_dir}/{self.s3.sub_dir}"):
                os.makedirs(f"{self.target_dir}/{self.s3.sub_dir}")
            with open(
                f"{self.target_dir}/{self.s3.sub_dir}/{search_doc['gridfs_id']}", "wb",
            ) as outfile:
                outfile.write(data)
                data = None

        return search_doc

    def update_targets(self, items):
        """
        Update minio index store with metadata

        Args:
            items ([([dict],[int])]): A list of tuples of materials to update and the corresponding processed task_ids
        """
        items = list(filter(None, items))

        if len(items) > 0:
            self.logger.info("Uploading {} band structures".format(len(items)))
            self.s3.update(items, key=[key for key in items[0] if key != "data"])
        else:
            self.logger.info("No band structure to copy")


def calc_etag(input_data):
    m = md5()
    chunk_size = 1024 * 1024
    for i in range(0, len(input_data), chunk_size):
        m.update(input_data[i : i + chunk_size])
    return m.hexdigest()

