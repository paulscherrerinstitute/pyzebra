import subprocess
import xml.etree.ElementTree as ET


REFLECTION_PRINTER_FORMATS = (
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
)

ALGORITHMS = ("adaptivemaxcog", "adaptivedynamic")


def anatric(config_file):
    subprocess.run(["anatric", config_file], check=True)


class AnatricConfig:
    def __init__(self, filename=None):
        if filename:
            self.load_from_file(filename)

    def load_from_file(self, filename):
        self._tree = ET.parse(filename)
        self._root = self._tree.getroot()

        self._alg_elems = dict()
        for alg in ALGORITHMS:
            self._alg_elems[alg] = ET.Element("Algorithm", attrib={"implementation": alg})

        self._alg_elems[self.algorithm] = self._tree.find("Algorithm")

    def save_as(self, filename):
        self._tree.write(filename)

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

        self._tree.find("FileList").tag = tag

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
        range_vals = self._filelist_elem.find("range").attrib
        return (int(range_vals["start"]), int(range_vals["end"]))

    @filelist_ranges.setter
    def filelist_ranges(self, value):
        range_vals = self._filelist_elem.find("range").attrib
        range_vals["start"] = str(value[0])
        range_vals["end"] = str(value[1])

    @property
    def crystal_sample(self):
        return self._tree.find("crystal").find("Sample").attrib["name"]

    @crystal_sample.setter
    def crystal_sample(self, value):
        self._tree.find("crystal").find("Sample").attrib["name"] = value

    @property
    def crystal_lambda(self):
        elem = self._tree.find("crystal").find("lambda")
        if elem is not None:
            return elem.attrib["value"]
        return None

    @crystal_lambda.setter
    def crystal_lambda(self, value):
        self._tree.find("crystal").find("lambda").attrib["value"] = value

    @property
    def crystal_zeroOM(self):
        elem = self._tree.find("crystal").find("zeroOM")
        if elem is not None:
            return elem.attrib["value"]
        return None

    @crystal_zeroOM.setter
    def crystal_zeroOM(self, value):
        self._tree.find("crystal").find("zeroOM").attrib["value"] = value

    @property
    def crystal_zeroSTT(self):
        elem = self._tree.find("crystal").find("zeroSTT")
        if elem is not None:
            return elem.attrib["value"]
        return None

    @crystal_zeroSTT.setter
    def crystal_zeroSTT(self, value):
        self._tree.find("crystal").find("zeroSTT").attrib["value"] = value

    @property
    def crystal_zeroCHI(self):
        elem = self._tree.find("crystal").find("zeroCHI")
        if elem is not None:
            return elem.attrib["value"]
        return None

    @crystal_zeroCHI.setter
    def crystal_zeroCHI(self, value):
        self._tree.find("crystal").find("zeroCHI").attrib["value"] = value

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
    def dist1(self):
        return self._tree.find("DataFactory").find("dist1").attrib["value"]

    @dist1.setter
    def dist1(self, value):
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

        self._root.remove(self._tree.find("Algorithm"))
        self._root.append(self._alg_elems[value])

    def _get_alg_attr(self, alg, tag, attr):
        param_elem = self._alg_elems[alg].find(tag)
        if param_elem is None:
            return None
        return param_elem.attrib[attr]

    # --- adaptivemaxcog
    @property
    def threshold(self):
        return self._get_alg_attr("adaptivemaxcog", "threshold", "value")

    @threshold.setter
    def threshold(self, value):
        alg_elem = self._alg_elems["adaptivemaxcog"]
        param_elem = alg_elem.find("threshold")
        if param_elem is None:
            alg_elem.append(ET.Element("threshold", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def shell(self):
        return self._get_alg_attr("adaptivemaxcog", "shell", "value")

    @shell.setter
    def shell(self, value):
        alg_elem = self._alg_elems["adaptivemaxcog"]
        param_elem = alg_elem.find("shell")
        if param_elem is None:
            alg_elem.append(ET.Element("shell", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def steepness(self):
        return self._get_alg_attr("adaptivemaxcog", "steepness", "value")

    @steepness.setter
    def steepness(self, value):
        alg_elem = self._alg_elems["adaptivemaxcog"]
        param_elem = alg_elem.find("steepness")
        if param_elem is None:
            alg_elem.append(ET.Element("steepness", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def duplicateDistance(self):
        return self._get_alg_attr("adaptivemaxcog", "duplicateDistance", "value")

    @duplicateDistance.setter
    def duplicateDistance(self, value):
        alg_elem = self._alg_elems["adaptivemaxcog"]
        param_elem = alg_elem.find("duplicateDistance")
        if param_elem is None:
            alg_elem.append(ET.Element("duplicateDistance", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def maxequal(self):
        return self._get_alg_attr("adaptivemaxcog", "maxequal", "value")

    @maxequal.setter
    def maxequal(self, value):
        alg_elem = self._alg_elems["adaptivemaxcog"]
        param_elem = alg_elem.find("maxequal")
        if param_elem is None:
            alg_elem.append(ET.Element("maxequal", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def aps_window(self):
        res = dict()
        for coord in ("x", "y", "z"):
            res[coord] = self._get_alg_attr("adaptivemaxcog", "window", coord)
        return res

    @aps_window.setter
    def aps_window(self, value):
        alg_elem = self._alg_elems["adaptivemaxcog"]
        param_elem = alg_elem.find("window")
        if param_elem is None:
            alg_elem.append(ET.Element("window", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    # --- adaptivedynamic
    @property
    def adm_window(self):
        res = dict()
        for coord in ("x", "y", "z"):
            res[coord] = self._get_alg_attr("adaptivedynamic", "window", coord)
        return res

    @adm_window.setter
    def adm_window(self, value):
        alg_elem = self._alg_elems["adaptivedynamic"]
        param_elem = alg_elem.find("window")
        if param_elem is None:
            alg_elem.append(ET.Element("window", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def border(self):
        res = dict()
        for coord in ("x", "y", "z"):
            res[coord] = self._get_alg_attr("adaptivedynamic", "border", coord)
        return res

    @border.setter
    def border(self, value):
        alg_elem = self._alg_elems["adaptivedynamic"]
        param_elem = alg_elem.find("border")
        if param_elem is None:
            alg_elem.append(ET.Element("border", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def minWindow(self):
        res = dict()
        for coord in ("x", "y", "z"):
            res[coord] = self._get_alg_attr("adaptivedynamic", "minWindow", coord)
        return res

    @minWindow.setter
    def minWindow(self, value):
        alg_elem = self._alg_elems["adaptivedynamic"]
        param_elem = alg_elem.find("minWindow")
        if param_elem is None:
            alg_elem.append(ET.Element("minWindow", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def reflectionFile(self):
        return self._get_alg_attr("adaptivedynamic", "reflectionFile", "filename")

    @reflectionFile.setter
    def reflectionFile(self, value):
        alg_elem = self._alg_elems["adaptivedynamic"]
        param_elem = alg_elem.find("reflectionFile")
        if param_elem is None:
            alg_elem.append(ET.Element("reflectionFile", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def targetMonitor(self):
        return self._get_alg_attr("adaptivedynamic", "targetMonitor", "value")

    @targetMonitor.setter
    def targetMonitor(self, value):
        alg_elem = self._alg_elems["adaptivedynamic"]
        param_elem = alg_elem.find("targetMonitor")
        if param_elem is None:
            alg_elem.append(ET.Element("targetMonitor", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def smoothSize(self):
        return self._get_alg_attr("adaptivedynamic", "smoothSize", "value")

    @smoothSize.setter
    def smoothSize(self, value):
        alg_elem = self._alg_elems["adaptivedynamic"]
        param_elem = alg_elem.find("smoothSize")
        if param_elem is None:
            alg_elem.append(ET.Element("smoothSize", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def loop(self):
        return self._get_alg_attr("adaptivedynamic", "loop", "value")

    @loop.setter
    def loop(self, value):
        alg_elem = self._alg_elems["adaptivedynamic"]
        param_elem = alg_elem.find("loop")
        if param_elem is None:
            alg_elem.append(ET.Element("loop", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def minPeakCount(self):
        return self._get_alg_attr("adaptivedynamic", "minPeakCount", "value")

    @minPeakCount.setter
    def minPeakCount(self, value):
        alg_elem = self._alg_elems["adaptivedynamic"]
        param_elem = alg_elem.find("minPeakCount")
        if param_elem is None:
            alg_elem.append(ET.Element("minPeakCount", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value

    @property
    def displacementCurve(self):
        param_elem = self._alg_elems["adaptivedynamic"].find("displacementCurve")
        if param_elem is None:
            return None
        return param_elem.attrib["value"]

    @displacementCurve.setter
    def displacementCurve(self, value):
        alg_elem = self._alg_elems["adaptivedynamic"]
        param_elem = alg_elem.find("displacementCurve")
        if param_elem is None:
            alg_elem.append(ET.Element("displacementCurve", attrib={"value": value}))
        else:
            param_elem.attrib["value"] = value
