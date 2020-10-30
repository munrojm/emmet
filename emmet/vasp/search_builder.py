from maggma.builders import Builder
from collections import defaultdict


class SearchBuilder(Builder):
    def __init__(
        self,
        materials,
        thermo,
        xas,
        grain_boundaries,
        bandstructure,
        dos,
        magnetism,
        elasticity,
        dielectric,
        phonon,
        substrates,
        surfaces,
        eos,
        search,
        query=None,
        **kwargs
    ):

        self.materials = materials
        self.thermo = thermo
        self.xas = xas
        self.grain_boundaries = grain_boundaries
        self.bandstructure = bandstructure
        self.dos = dos
        self.magnetism = magnetism
        self.elasticity = elasticity
        self.dielectric = dielectric
        self.phonon = phonon
        self.substrates = substrates
        self.surfaces = surfaces
        self.eos = eos
        self.search = search
        self.query = query if query else {}

        super().__init__(
            sources=[
                materials,
                thermo,
                xas,
                grain_boundaries,
                bandstructure,
                dos,
                magnetism,
                elasticity,
                dielectric,
                phonon,
                surfaces,
                substrates,
                eos,
            ],
            targets=[search],
            **kwargs
        )

        self.chunk_size = 1

    def get_items(self):
        """
        Gets all items to process

        Returns:
            list of relevant materials and data
        """

        self.logger.info("Search Builder Started")

        q = dict(self.query)

        sbox_q = {"_sbxn": "core"}

        mat_ids = self.materials.distinct(field="task_id", criteria={**q, **sbox_q})
        search_ids = self.search.distinct(field="task_id", criteria=q)
        thermo_ids = self.thermo.distinct(field="task_id", criteria=q)

        search_set = set(mat_ids).intersection(thermo_ids) - set(search_ids)

        search_list = [key for key in search_set]

        split_size = 200

        chunk_list = [
            search_list[i : i + split_size]
            for i in range(0, len(search_list), split_size)
        ]
        self.total = len(chunk_list)

        self.logger.debug(
            "Processing {} materials in {} chunks".format(
                len(search_list), len(chunk_list)
            )
        )

        for entry in chunk_list:

            query = {"$in": entry}

            data = {
                "materials": list(self.materials.query({self.materials.key: query})),
                "thermo": list(self.thermo.query({self.thermo.key: query})),
                "xas": list(self.xas.query({self.xas.key: query})),
                "grain_boundaries": list(
                    self.grain_boundaries.query({self.grain_boundaries.key: query})
                ),
                "bandstructure": list(
                    self.bandstructure.query({self.bandstructure.key: query})
                ),
                "dos": list(self.dos.query({self.dos.key: query})),
                "magnetism": list(self.magnetism.query({self.magnetism.key: query})),
                "elasticity": list(self.elasticity.query({self.elasticity.key: query})),
                "dielectric": list(self.dielectric.query({self.dielectric.key: query})),
                "phonon": list(
                    self.phonon.query({self.phonon.key: query}, [self.phonon.key])
                ),
                "surface_properties": list(
                    self.surfaces.query({self.surfaces.key: query})
                ),
                "substrates": list(self.surfaces.query({self.substrates.key: query})),
                "eos": list(self.eos.query({self.eos.key: query}, [self.eos.key])),
            }

            yield data

    def process_item(self, item):

        d = defaultdict(dict)

        # Materials

        materials_fields = [
            "nsites",
            "elements",
            "nelements",
            "composition",
            "composition_reduced",
            "formula_pretty",
            "formula_anonymous",
            "chemsys",
            "volume",
            "density",
            "symmetry",
            "task_id",
            "structure",
            "deprecated",
        ]

        ids = []

        for doc in item["materials"]:

            ids.append(doc[self.materials.key])
            d[ids[-1]] = defaultdict(dict)
            for field in materials_fields:
                d[ids[-1]][field] = doc[field]

        for id in ids:

            d[id]["has_props"] = []

            # Thermo

            thermo_fields = [
                "energy",
                "energy_per_atom",
                "formation_energy_per_atom",
                "e_above_hull",
                "is_stable",
            ]

            for doc in item["thermo"]:
                if doc[self.thermo.key] == id:
                    for field in thermo_fields:
                        d[id][field] = doc["thermo"][field]

                    d[id]["corrected_energy"] = doc["thermo"]["explanation"][
                        "corrected_energy"
                    ]

                    try:
                        d[id]["eq_reaction_e"] = doc["thermo"]["eq_reaction_e"]
                    except KeyError:
                        pass

            # XAS

            xas_fields = ["absorbing_element", "edge", "spectrum_type", "xas_id"]

            for doc in item["xas"]:
                if doc[self.xas.key] == id:

                    if d[id].get("xas", None) is None:
                        d[id]["xas"] = []

                    d[id]["has_props"].append("xas")

                    d[id]["xas"].append({field: doc[field] for field in xas_fields})

            # GB

            gb_fields = ["gb_energy", "sigma", "type", "rotation_angle", "w_sep"]

            for doc in item["grain_boundaries"]:
                if doc[self.grain_boundaries.key] == id:

                    if d[id].get("grain_boundaries", None) is None:
                        d[id]["grain_boundaries"] = []

                    d[id]["has_props"].append("grain_boundaries")

                    d[id]["grain_boundaries"].append(
                        {field: doc[field] for field in gb_fields}
                    )

            # Bandstructure

            bandstructure_fields = [
                "energy",
                "direct",
            ]

            for doc in item["bandstructure"]:
                if doc[self.bandstructure.key] == id:
                    d[id]["has_props"].append("bandstructure")
                    for type in ["sc", "hin", "lm"]:
                        for field in bandstructure_fields:
                            e = doc[type].get("band_gap", None).get(field, None)
                            if e is not None:
                                d[id]["{}_{}".format(type, field)] = e
            # DOS

            for doc in item["dos"]:
                if doc[self.dos.key] == id:
                    d[id]["has_props"].append("dos")

                    dos_bgap = doc["total"]["band_gap"]

                    for entry in dos_bgap:
                        if int(entry) == -1:
                            spin = "down"
                        else:
                            spin = "up"

                        d[id]["dos_energy_{}".format(spin)] = dos_bgap[entry]

            # Magnetism

            magnetism_fields = [
                "ordering",
                "total_magnetization",
                "total_magnetization_normalized_vol",
                "total_magnetization_normalized_formula_units",
            ]

            d[id]["spin_polarized"] = False

            for doc in item["magnetism"]:
                if doc[self.magnetism.key] == id:
                    d[id]["has_props"].append("magnetism")

                    d[id]["spin_polarized"] = True

                    for field in magnetism_fields:
                        d[id][field] = doc["magnetism"][field]

            # Elasticity

            elasticity_fields = [
                "k_voigt",
                "k_reuss",
                "k_vrh",
                "g_voigt",
                "g_reuss",
                "g_vrh",
                "universal_anisotropy",
                "homogeneous_poisson",
            ]

            for doc in item["elasticity"]:
                if doc[self.elasticity.key] == id:
                    d[id]["has_props"].append("elasticity")

                    for field in elasticity_fields:
                        d[id][field] = doc["elasticity"][field]

            # Dielectric and Piezo

            dielectric_fields = [
                "e_total",
                "e_ionic",
                "e_static",
                "n",
            ]

            piezo_fields = ["e_ij_max"]

            for doc in item["dielectric"]:
                if doc[self.dielectric.key] == id:
                    check_dielectric = doc.get("dielectric", None)
                    check_piezo = doc.get("piezo", None)
                    if check_dielectric is not None and check_dielectric != {}:
                        d[id]["has_props"].append("dielectric")

                        for field in dielectric_fields:
                            d[id][field] = doc["dielectric"][field]

                    if check_piezo is not None and check_piezo != {}:
                        d[id]["has_props"].append("piezoelectric")

                        for field in piezo_fields:
                            d[id][field] = doc["piezo"][field]

            # Surface properties

            surface_fields = [
                "weighted_surface_energy",
                "weighted_surface_energy_EV_PER_ANG2",
                "shape_factor",
                "surface_anisotropy",
            ]

            for doc in item["surface_properties"]:
                if doc[self.surfaces.key] == id:
                    d[id]["has_props"].append("surface_properties")

                    for field in surface_fields:
                        d[id][field] = doc[field]

            # EOS

            for doc in item["eos"]:

                if doc[self.eos.key] == id:
                    d[id]["has_props"].append("eos")

            d[id]["has_props"] = list(set(d[id]["has_props"]))

            # Phonon

            for doc in item["phonon"]:

                if doc[self.phonon.key] == id:
                    d[id]["has_props"].append("phonon")

            d[id]["has_props"] = list(set(d[id]["has_props"]))

            # Substrates

            for doc in item["substrates"]:

                if doc[self.substrates.key] == id:
                    d[id]["has_props"].append("substrates")

            d[id]["has_props"] = list(set(d[id]["has_props"]))

        return d

    def update_targets(self, items):
        """
        Copy each seardh doc to the store

        Args:
            items ([dict]): A list of tuples of docs to update
        """
        items = list(filter(None, items))

        if len(items) > 0:
            self.logger.info(
                "Inserting {} search docs".format(len(list(items[0].keys())))
            )
            for key, doc in items[0].items():
                self.search.update(doc, key=self.search.key)
        else:
            self.logger.info("No search entries to copy")