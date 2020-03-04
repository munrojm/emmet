import json
import logging
import os
import unicodedata
import warnings
import re
from multiprocessing import Manager, Pool

from atomate.utils.utils import get_meta_from_structure
from monty.io import zopen
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.apps.borg.hive import AbstractDrone
from pymatgen.apps.borg.queen import BorgQueen
from pymatgen.io.cif import CifParser
from pymatgen import Composition, Element


logger = logging.getLogger(__name__)
logging.basicConfig(filename='icsd_to_mongo.log', level=logging.DEBUG)
logging.captureWarnings(capture=True)

# clear previous Error_Record and log
with open('Error_Record', 'w') as err_rec:
    err_rec.close()
with open('icsd_to_mongo.log', 'w') as log:
    log.close()


class IcsdDrone(AbstractDrone):

    def __init__(self):
        # filler
        self.field = 1

    def _assimilate_from_cif(self, cif_path):
        # capture any warnings generated by parsing cif file

        file_ID = cif_path.split('/')[-1].split(".")[0]

        cif_meta = {}
        with warnings.catch_warnings(record=True) as w:
            cif_parser = CifParser(cif_path)
            for warn in w:
                if 'cifwarnings' in cif_meta:
                    cif_meta['cifwarnings'].append(str(warn.message))
                else:
                    cif_meta['cifwarnings'] = [str(warn.message)]
                logger.warning('{}: {}'.format(file_ID, warn.message))

        cif_dict = cif_parser.as_dict()
        orig_id = list(cif_dict.keys())[0]
        easy_dict = cif_dict[orig_id]

        if '_chemical_name_mineral' in easy_dict:
            cif_meta['min_name'] = easy_dict['_chemical_name_mineral']
        if '_chemical_name_systematic' in easy_dict:
            cif_meta['chem_name'] = easy_dict['_chemical_name_systematic']
        if '_cell_measurement_pressure' in easy_dict:
            cif_meta['pressure'] = float(
                easy_dict['_cell_measurement_pressure'])/1000
        else:
            cif_meta['pressure'] = .101325

        with warnings.catch_warnings(record=True) as w:
            try:
                struc = cif_parser.get_structures()[0]
            except ValueError as err:
                # if cif parsing raises error, write icsd_id to Error_Record and do NOT add structure to mongo database
                logger.error(file_ID + ': {}'.format(err) +
                             "\nDid not insert structure into Mongo Collection")
                with open('Error_Record', 'a') as err_rec:
                    err_rec.write(str(file_ID)+': {}\n'.format(err))
                    err_rec.close()
            else:
                references = self.bibtex_from_cif(cif_path)
                history = [{'name': 'ICSD', 'url': 'https://icsd.fiz-karlsruhe.de/',
                            'description': {'id': file_ID}}]

                cif_meta['references'] = references
                cif_meta['history'] = history

                atomate_meta = get_meta_from_structure(struc)
                # data['nsites'] = meta['nsites']
                # data['elements'] = meta['elements']
                # data['nelements'] = meta['nelements']
                # data['formula'] = meta['formula']
                # data['formula_reduced'] = meta['formula_pretty']
                # data['formula_reduced_abc'] = meta['formula_reduced_abc']
                # data['formula_anonymous'] = meta['formula_anonymous']
                # data['chemsys'] = meta['chemsys']
                # data['is_valid'] = meta['is_valid']
                # data['is_ordered'] = meta['is_ordered']

            # unfortunately any warnings are logged after any errors. Not too big of an issue
            for warn in w:
                if 'cifwarnings' in cif_meta:
                    cif_meta['cifwarnings'].append(str(warn.message))
                else:
                    cif_meta['cifwarnings'] = [str(warn.message)]
                logger.warning('{}: {}'.format(file_ID, warn.message))

        return(struc, cif_meta, atomate_meta)

    def assimilate(self, path):
        """
        Assimilate data in a directory path into a pymatgen object. Because of
        the quirky nature of Python"s multiprocessing, the object must support
        pymatgen's as_dict() for parallel processing.
        Args:
            path: directory path
        Returns:
            An assimilated object
        """

        files = os.listdir(path)
        file_ID = path.split('/')[-1]
        print(file_ID)

        cif_path = os.path.join(path, file_ID + '.cif')

        struc, cifmetadata, atomate_meta = self._assimilate_from_cif(cif_path)

        json_path = os.path.join(path, file_ID + '.json')

        metadata = {}
        if os.path.exists(json_path):
            metadata = self._assimilate_from_crawling(json_path)

            icsd_c = Composition(metadata['chemical_formula']).remove_charges().reduced_composition
            cif_c = struc.composition.remove_charges().reduced_composition

            metadata['consistent_composition'] = cif_c.almost_equals(icsd_c)

            metadata['implicit_hydrogen'] = self._has_implicit_H(icsd_c, cif_c)

            deuterium_indices = [ind for ind, s in enumerate(
                struc.sites) if re.findall(r'[A-z]+', s.species_string)[0] == "D"]
            tritium_indices = [ind for ind, s in enumerate(
                struc.sites) if re.findall(r'[A-z]+', s.species_string)[0] == "T"]

            for i_H in deuterium_indices + tritium_indices:
                struc.replace(i_H, "H")

            metadata['deuterium_indices'] = deuterium_indices
            metadata['tritium_indices'] = tritium_indices

        data = {
            'structure': struc,
            'metadata': metadata,
            "cifmetadata": cifmetadata
        }

        for key, val in atomate_meta.items():
            data[key] = val

        return(data)

    def _has_implicit_H(self, icsd_comp, cif_comp):
        icsd = icsd_comp.as_dict()
        cif = cif_comp.as_dict()

        if 'H' in icsd:
            if 'H' in cif:
                # Tolerance of 0.1 is derived from
                # almost_equals in pymatgen's Composition
                if abs(icsd['H'] - cif['H']) > 0.1:
                    return(True)

            else:
                return(True)

        return(False)

    def _assimilate_from_crawling(self, json_path):
        with open(json_path) as f:
            metadata = json.load(f)

        refs = []
        for key in ['reference', 'reference_1', 'reference_2', 'reference_3']:
            refs.append(metadata.pop(key))

        refs = list(set(refs))
        refs.remove("")

        metadata['references'] = refs

        return(metadata)

    def get_valid_paths(self, path):
        """
        Checks if path contains valid data for assimilation, and then returns
        the valid paths. The paths returned can be a list of directory or file
        paths, depending on what kind of data you are assimilating. For
        example, if you are assimilating VASP runs, you are only interested in
        directories containing vasprun.xml files. On the other hand, if you are
        interested converting all POSCARs in a directory tree to cifs for
        example, you will want the file paths.
        Args:
            path: input path as a tuple generated from os.walk, i.e.,
                (parent, subdirs, files).
        Returns:
            List of valid dir/file paths for assimilation
        """
        (parent, subdirs, files) = path
        if len(subdirs) != 0:
            return [os.path.join(parent, dir_name) for dir_name in subdirs]

        else:
            return []
        return []

    def bibtex_from_cif(self, cif_string):
            # if input is a cif filename read from file, else assume input is cif string
        if cif_string.endswith(".cif"):
            cif_dict = CifParser(cif_string).as_dict()
        else:
            cif_dict = CifParser.from_string(cif_string).as_dict()

        orig_id = list(cif_dict.keys())[0]

        # more accesable dict
        easy_dict = cif_dict[orig_id]

        # generate bibTex string
        bibtex_str = "@article{"

        # use first author's last name as key + year. not sure about this
        bibtex_key = easy_dict['_publ_author_name'][0].replace(' ', '')
        bibtex_key = bibtex_key[0:bibtex_key.find(',')]
        bibtex_key += easy_dict['_citation_year'][0]
        bibtex_str += bibtex_key + ",\n"

        # add title
        bibtex_str += "title = {" + easy_dict['_publ_section_title'] + "},\n"

        # add authors
        bibtex_str += "author = {" + \
            " and ".join(easy_dict['_publ_author_name']) + "},\n"

        # add journal title
        bibtex_str += "journal = {" + \
            easy_dict['_citation_journal_full'][0] + "},\n"

        # add year
        bibtex_str += "year = {" + easy_dict['_citation_year'][0] + "},\n"

        # add volume number
        bibtex_str += "volume = {" + \
            easy_dict['_citation_journal_volume'][0] + "},\n"

        # add pages
        bibtex_str += "pages = {" + easy_dict['_citation_page_first'][0] + \
            "-" + easy_dict['_citation_page_last'][0] + "},\n"

        # add ASTM id
        bibtex_str += "ASTM_id = {" + \
            easy_dict['_citation_journal_id_ASTM'][0] + "},\n"

        # end string and normalize to ascii
        bibtex_str += "}"
        #bibtex_str = unicodedata.normalize('NFKD', bibtex_str).encode('ascii','ignore')

        # print(bibtex_str)

        return bibtex_str


class IcsdQueen(BorgQueen):

    def __init__(self, drone, rootpath=None, number_of_drones=1):
        self._drone = drone
        self._num_drones = number_of_drones
        self._data = []

        if rootpath:
            if number_of_drones > 1:
                self.parallel_assimilate(rootpath)
            else:
                self.serial_assimilate(rootpath)

    def parallel_assimilate(self, rootpath):
        """
        Assimilate the entire subdirectory structure in rootpath.
        """
        logger.info('Scanning for valid paths...')
        valid_paths = []
        for (parent, subdirs, files) in os.walk(rootpath):
            valid_paths.extend(self._drone.get_valid_paths(self._drone, (parent, subdirs,
                                                                         files)))
        manager = Manager()
        data = manager.list()
        status = manager.dict()
        status['count'] = 0
        status['total'] = len(valid_paths)
        logger.info('{} valid paths found.'.format(len(valid_paths)))
        p = Pool(self._num_drones)
        p.map(self.order_assimilation, ((path, self._drone, data, status)
                                        for path in valid_paths))
        for d in data:
            self._data.append(json.loads(d, cls=MontyDecoder))

    def serial_assimilate(self, rootpath):
        """
        Assimilate the entire subdirectory structure in rootpath serially.
        """
        valid_paths = []
        for (parent, subdirs, files) in os.walk(rootpath):
            valid_paths.extend(self._drone.get_valid_paths(self._drone, (parent, subdirs,
                                                                         files)))
        data = []
        count = 0
        total = len(valid_paths)
        for path in valid_paths:
            newdata = self._drone.assimilate(self._drone, path)
            self._data.append(newdata)
            count += 1
            logger.info('{}/{} ({:.2f}%) done'.format(count, total,
                                                      count / total * 100))
        for d in data:
            self._data.append(json.loads(d, cls=MontyDecoder))

    def order_assimilation(self, args):
        """
        Internal helper method for BorgQueen to process assimilation
        """
        (path, drone, data, status) = args
        newdata = drone.assimilate(drone, path)
        if newdata:
            data.append(json.dumps(newdata, cls=MontyEncoder))
        status['count'] += 1
        count = status['count']
        total = status['total']
        logger.info('{}/{} ({:.2f}%) done'.format(count, total,
                                                  count / total * 100))


if __name__ == '__main__':
    path_to_dirs = os.getcwd()
    IcsdQueen(IcsdDrone, rootpath=path_to_dirs, number_of_drones=1)