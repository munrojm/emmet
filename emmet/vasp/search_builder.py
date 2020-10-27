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
        electronic_structure,
        magnetism,
        elasticity,
        dielectric,
        phonon,
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
        self.electronic_structure = electronic_structure
        self.magnetism = magnetism
        self.elasticity = elasticity
        self.dielectric = dielectric
        self.phonon = phonon
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
                eos,
            ],
            targets=[search],
            **kwargs
        )

        self.chunk_size = 5

    def get_items(self):
        """
        Gets all items to process

        Returns:
            list of relevant materials and data
        """

        self.logger.info("Search Builder Started")

        q = dict(self.query)

        mat_ids = self.materials.distinct(field="material_id", criteria=q)
        search_ids = self.search.distinct(field="material_id", criteria=q)

        search_set = set(mat_ids) - set(search_ids)

        search_list = [key for key in search_set]

        self.logger.debug("Processing {} materials".format(len(search_list)))

        self.total = len(search_list)

        for entry in search_list:

            data = {
                "materials": self.materials.query_one({"material_id": entry}),
                "thermo": self.thermo.query_one({"material_id": entry}),
                "xas": list(self.xas.query({"task_id": entry})),
                "grain_boundaries": list(
                    self.grain_boundaries.query({"task_id": entry})
                ),
                "bandstructure": self.bandstructure.query_one({"task_id": entry}),
                "dos": self.dos.query_one({"task_id": entry}),
                "magnetism": self.magnetism.query({"task_id": entry}),
                "elasticity": self.elasticity.query({"task_id": entry}),
                "dielectric": self.dielectric.query({"task_id": entry}),
                "phonon": self.phonon.query({"task_id": entry}),
                "surface_properties": self.surfaces.query_one({"task_id": entry}),
                "eos": self.eos.query_one({"task_id": entry}, ["task_id"]),
            }

            yield data

    def process_item(self, item):

        d = {}
        d["has_props"] = []

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
            "density_atomic",
            "symmetry",
            "material_id",
            "structure",
        ]

        for field in materials_fields:
            d[field] = item["materials"][field]

        # Thermo

        thermo_fields = [
            "energy",
            "energy_per_atom",
            "formation_energy_per_atom",
            "e_above_hull",
            "is_stable",
        ]

        for field in thermo_fields:
            d[field] = item["thermo"][field]

        # XAS

        xas_fields = ["absorbing_element", "edge", "spectrum_type", "xas_id"]

        if item["xas"] != {}:
            d["has_props"].append("xas")
            d["xas"] = []

            for doc in item["xas"]:
                d["xas"].append({field: doc[field] for field in xas_fields})

        # GB

        gb_fields = ["gb_energy", "sigma", "type", "rotation_angle", "w_sep"]

        if item["grain_boundaries"] != {}:
            d["has_props"].append("grain_boundaries")
            d["grain_boundaries"] = []

            for doc in item["grain_boundaries"]:
                d["grain_boundaries"].append({field: doc[field] for field in gb_fields})

        # Bandstructure

        bandstructure_fields = [
            "energy",
            "direct",
        ]

        if item["bandstructure"] != {}:
            d["has_props"].append("bandstructure")
            for type in ["sc", "hin", "lm"]:
                for field in bandstructure_fields:
                    d["{}_{}".format(type, field)] = item["bandstructure"][type][
                        "band_gap"
                    ][field]

        # DOS

        if item["dos"] != {}:
            d["has_props"].append("dos")

            dos_bgap = item["dos"]["total"]["band_gap"]

            for entry in dos_bgap:
                d["dos_energy_{}".format(entry)] = dos_bgap[entry]
                if int(entry) == -1:
                    d["spin_polarized"] = True

        # Elasticity

        elasticity_fields = [
            "k_voigt",
            "k_reuss",
            "k_vrh",
            "g_voigt",
            "g_reuss",
            "g_vrh",
            "universal_anisotropy",
            "homogenenous_poisson",
        ]

        if item["elasticity"] != {}:
            d["has_props"].append["elasticity"]

            for field in elasticity_fields:
                d[field] = item["elasticity"]["elasticity"][field]

        # Dielectric

        dielectric_fields = [
            "e_total",
            "e_ionic",
            "e_static",
            "n",
        ]

        if item["dielectric"] != {}:
            if item["dielectric"].get("dielectric", None) is not None:
                d["has_props"].append("dielectric")

                for field in dielectric_fields:
                    d[field] = item["dielectric"]["dielectric"][field]

        # Piezo

        piezo_fields = ["e_ij_max"]

        if item["dielectric"] != {}:
            if item["dielectric"].get("piezo", None) is not None:
                d["has_props"].append("piezoelectric")

                for field in piezo_fields:
                    d[field] = item["dielectric"]["piezo"][field]

        # Surface properties

        surface_fields = [
            "weighted_surface_energy",
            "weighted_surface_energy_EV_PER_ANG2",
            "shape_factor",
            "surface_anisotropy",
        ]

        if item["surfaces_properties"] != {}:
            d["has_props"].append("surface_properties")

            for field in surface_fields:
                d[field] = item["surface_properties"][field]

        # EOS

        if item["eos"] != {}:
            d["has_props"].append("eos")

        return d

    def update_targets(self, items):
        """
        Copy each seardh doc to the store

        Args:
            items ([dict]): A list of tuples of docs to update
        """
        items = list(filter(None, items))

        if len(items) > 0:
            self.logger.info("Inserting {} search docs".format(len(items)))

            self.search.update(items, key=self.search.key)
        else:
            self.logger.info("No search entries to copy")
