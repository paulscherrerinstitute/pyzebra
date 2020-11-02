from bokeh.application.handlers import Handler


class PyzebraHandler(Handler):
    """Provides a mechanism for generic bokeh applications to build up new streamvis documents.
    """

    def __init__(self, anatric_path):
        """Initialize a pyzebra handler for bokeh applications.

        Args:
            args (Namespace): Command line parsed arguments.
        """
        super().__init__()  # no-op

        self.anatric_path = anatric_path

    def modify_document(self, doc):
        """Modify an application document with pyzebra specific features.

        Args:
            doc (Document) : A bokeh Document to update in-place

        Returns:
            Document
        """
        doc.title = "pyzebra"
        doc.anatric_path = self.anatric_path

        return doc
