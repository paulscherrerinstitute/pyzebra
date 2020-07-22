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

    # --- adaptivemaxcog
    @property
    def threshold(self):
        return self._alg_elems["adaptivemaxcog"].find("threshold").attrib["value"]

    @threshold.setter
    def threshold(self, value):
        self._alg_elems["adaptivemaxcog"].find("threshold").attrib["value"] = value

    @property
    def shell(self):
        return self._alg_elems["adaptivemaxcog"].find("shell").attrib["value"]

    @shell.setter
    def shell(self, value):
        self._alg_elems["adaptivemaxcog"].find("shell").attrib["value"] = value

    @property
    def steepness(self):
        return self._alg_elems["adaptivemaxcog"].find("steepness").attrib["value"]

    @steepness.setter
    def steepness(self, value):
        self._alg_elems["adaptivemaxcog"].find("steepness").attrib["value"] = value

    @property
    def duplicateDistance(self):
        return self._alg_elems["adaptivemaxcog"].find("duplicateDistance").attrib["value"]

    @duplicateDistance.setter
    def duplicateDistance(self, value):
        self._alg_elems["adaptivemaxcog"].find("duplicateDistance").attrib["value"] = value

    @property
    def maxequal(self):
        return self._alg_elems["adaptivemaxcog"].find("maxequal").attrib["value"]

    @maxequal.setter
    def maxequal(self, value):
        self._alg_elems["adaptivemaxcog"].find("maxequal").attrib["value"] = value

    @property
    def aps_window(self):
        return self._alg_elems["adaptivemaxcog"].find("window").attrib["value"]

    @aps_window.setter
    def aps_window(self, value):
        self._alg_elems["adaptivemaxcog"].find("window").attrib["value"] = value

    # --- adaptivedynamic
    @property
    def adm_window(self):
        return self._alg_elems["adaptivedynamic"].find("window").attrib["value"]

    @adm_window.setter
    def adm_window(self, value):
        self._alg_elems["adaptivedynamic"].find("window").attrib["value"] = value

    @property
    def border(self):
        return self._alg_elems["adaptivedynamic"].find("border").attrib["value"]

    @border.setter
    def border(self, value):
        self._alg_elems["adaptivedynamic"].find("border").attrib["value"] = value

    @property
    def minWindow(self):
        return self._alg_elems["adaptivedynamic"].find("minWindow").attrib["value"]

    @minWindow.setter
    def minWindow(self, value):
        self._alg_elems["adaptivedynamic"].find("minWindow").attrib["value"] = value

    @property
    def reflectionFile(self):
        return self._alg_elems["adaptivedynamic"].find("reflectionFile").attrib["value"]

    @reflectionFile.setter
    def reflectionFile(self, value):
        self._alg_elems["adaptivedynamic"].find("reflectionFile").attrib["value"] = value

    @property
    def targetMonitor(self):
        return self._alg_elems["adaptivedynamic"].find("targetMonitor").attrib["value"]

    @targetMonitor.setter
    def targetMonitor(self, value):
        self._alg_elems["adaptivedynamic"].find("targetMonitor").attrib["value"] = value

    @property
    def smoothSize(self):
        return self._alg_elems["adaptivedynamic"].find("smoothSize").attrib["value"]

    @smoothSize.setter
    def smoothSize(self, value):
        self._alg_elems["adaptivedynamic"].find("smoothSize").attrib["value"] = value

    @property
    def loop(self):
        return self._alg_elems["adaptivedynamic"].find("loop").attrib["value"]

    @loop.setter
    def loop(self, value):
        self._alg_elems["adaptivedynamic"].find("loop").attrib["value"] = value

    @property
    def minPeakCount(self):
        return self._alg_elems["adaptivedynamic"].find("minPeakCount").attrib["value"]

    @minPeakCount.setter
    def minPeakCount(self, value):
        self._alg_elems["adaptivedynamic"].find("minPeakCount").attrib["value"] = value

    @property
    def displacementCurve(self):
        return self._alg_elems["adaptivedynamic"].find("displacementCurve").attrib["value"]

    @displacementCurve.setter
    def displacementCurve(self, value):
        self._alg_elems["adaptivedynamic"].find("displacementCurve").attrib["value"] = value
