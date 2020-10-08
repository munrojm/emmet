from maggma.builders import Builder

from pymatgen.electronic_structure.bandstructure import BandStructure

from ifermi.fermi_surface import FermiSurface
from ifermi.interpolator import Interpolater

# pyfftw can improve performance:
# pip install IFermi[decimation,smooth] pyfftw


class FermiSurfaceBuilder(Builder):
    def __init__(self, bandstructures, tasks, fermi, query=None, **kwargs):
        """
        Calculates fermi surfaces using uniform bandstructure objects

        bandstructures (Store) : store of bandstructures
        tasks (Store): tasks store to obtain supplmental data not found in
            bs object.
        fermi (Store) : store of fermi surfaces
        query (dict): dictionary to limit bandstructures to be analyzed
        """

        self.bandstructures = bandstructures
        self.tasks = tasks
        self.fermi = fermi
        self.query = query if query else {}

        super().__init__(sources=[bandstructures, tasks], targets=[fermi], **kwargs)

        self.chunk_size = 5

    def get_items(self):
        """
        Gets all items to process

        Returns:
            list relevant band structures
        """

        self.logger.info("Fermi Surface Builder Started")

        q = dict(self.query)

        # Obtain uniform band structure task_ids and fermi surface task_ids
        bs_ids = self.bandstructures.distinct(field="metadata.task_id", criteria=q)
        fermi_ids = self.fermi.distinct(field="task_id", criteria=q)

        bs_set = set(bs_ids) - set(fermi_ids)

        bs_list = [key for key in bs_set]

        self.logger.debug(
            "Processing {} uniform band structure objects".format(len(bs_list))
        )

        self.total = len(bs_list)

        for entry in bs_list:

            bs = self.bandstructures.query_one(criteria={"task_id": entry})

            if not bs.get("structure", None):
                bs["structure"] = self.tasks.query_one({"task_id": entry}, ["input"])[
                    "input"
                ]["structure"]

            metadata = self.bandstructures._files_store.query_one(
                criteria={"metadata.task_id": entry},
                properties=["_id", "metadata", "uploadDate"],
            )
            data = {
                "gridfs_id": str(metadata["_id"]),
                "task_id": str(metadata["metadata"]["task_id"]),
                self.fermi.last_updated_field: metadata["uploadDate"],
                "bs": bs,
            }

            yield data

    def process_item(self, item):

        # here item is a uniform band structure,
        # pymatgen.electronic_structure.bandstructure.BandStructure
        # if not a MapBuilder, would have to include a task_id in item too

        bs = BandStructure.from_dict(item["bs"])

        try:

            interpolater = Interpolater(bs)

            # interpolate factor has to be at least 2, preferrably above 5
            interpolate_factor = 8

            # energy cut-off reduces the work we have to do, should be > mu
            interp_bs, kpoint_dim = interpolater.interpolate_bands(
                interpolate_factor, energy_cutoff=1
            )

            mu_values = []
            mu_labels = {}
            if interp_bs.is_metal():
                mu_values.append(0)  # plot the Fermi surface
                mu_labels[0] = "Fermi surface"
            else:
                # for semiconductors, plot an isosurface slightly into
                # conduction/valence bands
                window = 0.1  # eV
                # efermi is set to be the centre of the gap in IFermi
                efermi = interp_bs.efermi
                # vbm
                vbm_mu = interp_bs.get_vbm()["energy"] - efermi - window
                mu_values.append(vbm_mu)
                mu_labels[vbm_mu] = "VBM"
                # cbm
                cbm_mu = interp_bs.get_cbm()["energy"] - efermi + window
                mu_values.append(cbm_mu)
                mu_labels[cbm_mu] = "CBM"

            fermi_surfaces = {}

            for mu in mu_values:

                # have to calculate twice, once to get the original number
                # of triangles so we can calculate the "decimation factor"
                fs = FermiSurface.from_band_structure(
                    interp_bs, kpoint_dim, mu=mu, wigner_seitz=True
                )

                total_verts = 0
                for spin, surfaces in fs.isosurfaces.items():
                    for surface in surfaces:
                        total_verts += len(surface[0])

                desired_max_verts = 10000
                decimation_factor = int(total_verts / desired_max_verts)

                fs = FermiSurface.from_band_structure(
                    interp_bs,
                    kpoint_dim,
                    mu=mu,
                    wigner_seitz=True,
                    smooth=True,
                    decimate_factor=decimation_factor,
                )

                fermi_surfaces[mu] = fs.as_dict()

            d = {
                "_".join(mu_labels[value].lower().split(" ")): fermi_surfaces[value]
                for value in mu_values
            }

            d.update(
                {
                    "task_id": item["task_id"],
                    self.fermi.last_updated_field: item[self.fermi.last_updated_field],
                    "gridfs_id": item["gridfs_id"],
                }
            )

        except Exception:
            d = None

        return d

    def update_targets(self, items):
        """
        Copy each fermi doc to the store

        Args:
            items ([dict]): A list of tuples of docs to update
        """
        items = list(filter(None, items))

        if len(items) > 0:
            self.logger.info("Inserting {} fermi surface docs".format(len(items)))

            self.fermi.update(items, key=self.fermi.key)
        else:
            self.logger.info("No dos entries to copy")
