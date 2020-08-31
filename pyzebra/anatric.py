import subprocess
import xml.etree.ElementTree as ET


ANATRIC_PATH = "/afs/psi.ch/project/sinq/rhel7/bin/anatric"
DATA_FACTORY_IMPLEMENTATION = [
    "trics",
    "morph",
    "d10",
]
REFLECTION_PRINTER_FORMATS = [
    "rafin",
    "rafinf",
    "rafin2d",
    "rafin2di",
    "orient",
    "shelx",
    "jana2k",
    "jana2kf",
    "raw",
    "oksana",
]

ALGORITHMS = ["adaptivemaxcog", "adaptivedynamic"]


def anatric(config_file):
    subprocess.run([ANATRIC_PATH, config_file], check=True)


class AnatricConfig:
    def __init__(self, filename=None):
        self._alg_elems = dict()
        for alg in ALGORITHMS:
            self._alg_elems[alg] = ET.Element("Algorithm", attrib={"implementation": alg})
            self._alg_elems[alg].text = "\n"
            self._alg_elems[alg].tail = "\n\n"

        root_elem = ET.Element("anatric")
        root_elem.text = "\n"
        root_elem.append(self._alg_elems[ALGORITHMS[0]])

        self._tree = ET.ElementTree(element=root_elem)

        if filename:
            self.load_from_file(filename)

    def load_from_file(self, filename):
        self._tree.parse(filename)
        self._alg_elems[self.algorithm] = self._tree.find("Algorithm")

    def save_as(self, filename):
        self._tree.write(filename)

    def _get_attr(self, name, tag, attr):
        elem = self._tree.find(name).find(tag)
        if elem is None:
            return None
        return elem.attrib[attr]

    def _set_attr(self, name, tag, attr, value):
        if value == "" or value is None:
            self._del_attr(name, tag)
            return

        tree_elem = self._tree.find(name)
        elem = tree_elem.find(tag)
        if elem is None:
            new_elem = ET.Element(tag, attrib={attr: value})
            new_elem.tail = "\n"
            tree_elem.append(new_elem)
        else:
            elem.attrib[attr] = value

    def _del_attr(self, name, tag):
        tree_elem = self._tree.find(name)
        param_elem = tree_elem.find(tag)
        if param_elem is not None:
            tree_elem.remove(param_elem)

    @property
    def logfile(self):
        return self._tree.find("logfile").attrib["file"]

    @logfile.setter
    def logfile(self, value):
        self._tree.find("logfile").attrib["file"] = value

    @property
    def logfile_verbosity(self):
        return self._tree.find("logfile").attrib["verbosity"]

    @logfile_verbosity.setter
    def logfile_verbosity(self, value):
        self._tree.find("logfile").attrib["verbosity"] = value

    @property
    def filelist_type(self):
        if self._tree.find("FileList") is not None:
            return "TRICS"
        return "SINQ"

    @filelist_type.setter
    def filelist_type(self, value):
        if value == "TRICS":
            tag = "FileList"
        elif value == "SINQ":
            tag = "SinqFileList"
        else:
            raise ValueError("FileList value can only by 'TRICS' or 'SINQ'")

        filelist_elem = self._tree.find("FileList") or self._tree.find("SinqFileList")
        filelist_elem.tag = tag

    @property
    def _filelist_elem(self):
        if self.filelist_type == "TRICS":
            filelist_elem = self._tree.find("FileList")
        else:  # SINQ
            filelist_elem = self._tree.find("SinqFileList")

        return filelist_elem

    @property
    def filelist_format(self):
        return self._filelist_elem.attrib["format"]

    @filelist_format.setter
    def filelist_format(self, value):
        self._filelist_elem.attrib["format"] = value

    @property
    def filelist_datapath(self):
        return self._filelist_elem.find("datapath").attrib["value"]

    @filelist_datapath.setter
    def filelist_datapath(self, value):
        self._filelist_elem.find("datapath").attrib["value"] = value

    @property
    def filelist_ranges(self):
        ranges = []
        for range_elem in self._filelist_elem.findall("range"):
            ranges.append((int(range_elem.attrib["start"]), int(range_elem.attrib["end"])))

        for range_elem in self._filelist_elem.findall("file"):
            ranges.append(int(range_elem.attrib["value"]))

        return ranges

    @filelist_ranges.setter
    def filelist_ranges(self, value):
        # clear old range elements
        filelist_elem = self._filelist_elem
        for range_elem in filelist_elem.findall("range"):
            filelist_elem.remove(range_elem)

        for range_elem in filelist_elem.findall("file"):
            filelist_elem.remove(range_elem)

        # add new range elements
        for range_vals in value:
            if len(range_vals) == 1:
                # single file
                tag = "file"
                attrib = {"value": range_vals[0]}
            else:
                # range of files
                tag = "range"
                attrib = {"start": range_vals[0], "end": range_vals[1]}

            range_elem = ET.Element(tag, attrib=attrib)
            range_elem.tail = "\n"
            filelist_elem.append(range_elem)

    @property
    def crystal_sample(self):
        return self._get_attr("crystal", "Sample", "name")

    @crystal_sample.setter
    def crystal_sample(self, value):
        self._set_attr("crystal", "Sample", "name", value)

    @property
    def crystal_lambda(self):
        return self._get_attr("crystal", "lambda", "value")

    @crystal_lambda.setter
    def crystal_lambda(self, value):
        self._set_attr("crystal", "lambda", "value", value)

    @property
    def crystal_zeroOM(self):
        return self._get_attr("crystal", "zeroOM", "value")

    @crystal_zeroOM.setter
    def crystal_zeroOM(self, value):
        self._set_attr("crystal", "zeroOM", "value", value)

    @property
    def crystal_zeroSTT(self):
        return self._get_attr("crystal", "zeroSTT", "value")

    @crystal_zeroSTT.setter
    def crystal_zeroSTT(self, value):
        self._set_attr("crystal", "zeroSTT", "value", value)

    @property
    def crystal_zeroCHI(self):
        return self._get_attr("crystal", "zeroCHI", "value")

    @crystal_zeroCHI.setter
    def crystal_zeroCHI(self, value):
        self._set_attr("crystal", "zeroCHI", "value", value)

    @property
    def crystal_UB(self):
        elem = self._tree.find("crystal").find("UB")
        if elem is not None:
            return elem.text
        return None

    @crystal_UB.setter
    def crystal_UB(self, value):
        self._tree.find("crystal").find("UB").text = value

    @property
    def dataFactory_implementation(self):
        return self._tree.find("DataFactory").attrib["implementation"]

    @dataFactory_implementation.setter
    def dataFactory_implementation(self, value):
        if value not in DATA_FACTORY_IMPLEMENTATION:
            raise ValueError("Unknown DataFactory implementation.")

        self._tree.find("DataFactory").attrib["implementation"] = value

    @property
    def dataFactory_dist1(self):
        return self._tree.find("DataFactory").find("dist1").attrib["value"]

    @dataFactory_dist1.setter
    def dataFactory_dist1(self, value):
        self._tree.find("DataFactory").find("dist1").attrib["value"] = value

    @property
    def reflectionPrinter_format(self):
        return self._tree.find("ReflectionPrinter").attrib["format"]

    @reflectionPrinter_format.setter
    def reflectionPrinter_format(self, value):
        if value not in REFLECTION_PRINTER_FORMATS:
            raise ValueError("Unknown ReflectionPrinter format.")

        self._tree.find("ReflectionPrinter").attrib["format"] = value

    @property
    def algorithm(self):
        return self._tree.find("Algorithm").attrib["implementation"]

    @algorithm.setter
    def algorithm(self, value):
        if value not in ALGORITHMS:
            raise ValueError("Unknown algorithm.")

        root = self._tree.getroot()
        root.remove(self._tree.find("Algorithm"))
        root.append(self._alg_elems[value])

    def _get_alg_attr(self, alg, tag, attr):
        param_elem = self._alg_elems[alg].find(tag)
        if param_elem is None:
            return None
        return param_elem.attrib[attr]

    def _set_alg_attr(self, alg, tag, attr, value):
        if value == "" or value is None:
            self._del_alg_attr(alg, tag)
            return

        alg_elem = self._alg_elems[alg]
        param_elem = alg_elem.find(tag)
        if param_elem is None:
            new_elem = ET.Element(tag, attrib={attr: value})
            new_elem.tail = "\n"
            alg_elem.append(new_elem)
        else:
            param_elem.attrib[attr] = value

    def _del_alg_attr(self, alg, tag):
        alg_elem = self._alg_elems[alg]
        param_elem = alg_elem.find(tag)
        if param_elem is not None:
            alg_elem.remove(param_elem)

    # --- adaptivemaxcog
    @property
    def threshold(self):
        return self._get_alg_attr("adaptivemaxcog", "threshold", "value")

    @threshold.setter
    def threshold(self, value):
        self._set_alg_attr("adaptivemaxcog", "threshold", "value", value)

    @property
    def shell(self):
        return self._get_alg_attr("adaptivemaxcog", "shell", "value")

    @shell.setter
    def shell(self, value):
        self._set_alg_attr("adaptivemaxcog", "shell", "value", value)

    @property
    def steepness(self):
        return self._get_alg_attr("adaptivemaxcog", "steepness", "value")

    @steepness.setter
    def steepness(self, value):
        self._set_alg_attr("adaptivemaxcog", "steepness", "value", value)

    @property
    def duplicateDistance(self):
        return self._get_alg_attr("adaptivemaxcog", "duplicateDistance", "value")

    @duplicateDistance.setter
    def duplicateDistance(self, value):
        self._set_alg_attr("adaptivemaxcog", "duplicateDistance", "value", value)

    @property
    def maxequal(self):
        return self._get_alg_attr("adaptivemaxcog", "maxequal", "value")

    @maxequal.setter
    def maxequal(self, value):
        self._set_alg_attr("adaptivemaxcog", "maxequal", "value", value)

    @property
    def aps_window(self):
        res = dict()
        for coord in ("x", "y", "z"):
            res[coord] = self._get_alg_attr("adaptivemaxcog", "window", coord)
        return res

    @aps_window.setter
    def aps_window(self, value):
        for coord in ("x", "y", "z"):
            self._set_alg_attr("adaptivemaxcog", "window", coord, value[coord])

    # --- adaptivedynamic
    @property
    def adm_window(self):
        res = dict()
        for coord in ("x", "y", "z"):
            res[coord] = self._get_alg_attr("adaptivedynamic", "window", coord)
        return res

    @adm_window.setter
    def adm_window(self, value):
        for coord in ("x", "y", "z"):
            self._set_alg_attr("adaptivedynamic", "window", coord, value[coord])

    @property
    def border(self):
        res = dict()
        for coord in ("x", "y", "z"):
            res[coord] = self._get_alg_attr("adaptivedynamic", "border", coord)
        return res

    @border.setter
    def border(self, value):
        for coord in ("x", "y", "z"):
            self._set_alg_attr("adaptivedynamic", "border", coord, value[coord])

    @property
    def minWindow(self):
        res = dict()
        for coord in ("x", "y", "z"):
            res[coord] = self._get_alg_attr("adaptivedynamic", "minWindow", coord)
        return res

    @minWindow.setter
    def minWindow(self, value):
        for coord in ("x", "y", "z"):
            self._set_alg_attr("adaptivedynamic", "minWindow", coord, value[coord])

    @property
    def reflectionFile(self):
        return self._get_alg_attr("adaptivedynamic", "reflectionFile", "filename")

    @reflectionFile.setter
    def reflectionFile(self, value):
        self._set_alg_attr("adaptivedynamic", "reflectionFile", "filename", value)

    @property
    def targetMonitor(self):
        return self._get_alg_attr("adaptivedynamic", "targetMonitor", "value")

    @targetMonitor.setter
    def targetMonitor(self, value):
        self._set_alg_attr("adaptivedynamic", "targetMonitor", "value", value)

    @property
    def smoothSize(self):
        return self._get_alg_attr("adaptivedynamic", "smoothSize", "value")

    @smoothSize.setter
    def smoothSize(self, value):
        self._set_alg_attr("adaptivedynamic", "smoothSize", "value", value)

    @property
    def loop(self):
        return self._get_alg_attr("adaptivedynamic", "loop", "value")

    @loop.setter
    def loop(self, value):
        self._set_alg_attr("adaptivedynamic", "loop", "value", value)

    @property
    def minPeakCount(self):
        return self._get_alg_attr("adaptivedynamic", "minPeakCount", "value")

    @minPeakCount.setter
    def minPeakCount(self, value):
        self._set_alg_attr("adaptivedynamic", "minPeakCount", "value", value)

    @property
    def displacementCurve(self):
        maps = []
        displacementCurve_elem = self._alg_elems["adaptivedynamic"].find("displacementCurve")
        if displacementCurve_elem is not None:
            for map_elem in displacementCurve_elem.findall("map"):
                maps.append(
                    (
                        float(map_elem.attrib["twotheta"]),
                        float(map_elem.attrib["x"]),
                        float(map_elem.attrib["y"]),
                    )
                )

        return maps

    @displacementCurve.setter
    def displacementCurve(self, value):
        # clear old map elements
        displacementCurve_elem = self._alg_elems["adaptivedynamic"].find("displacementCurve")
        for map_elem in displacementCurve_elem.findall("map"):
            displacementCurve_elem.remove(map_elem)

        # add new map elements
        for map_vals in value:
            attrib = {"twotheta": map_vals[0], "x": map_vals[1], "y": map_vals[2]}
            map_elem = ET.Element("map", attrib=attrib)
            map_elem.tail = "\n"
            displacementCurve_elem.append(map_elem)
