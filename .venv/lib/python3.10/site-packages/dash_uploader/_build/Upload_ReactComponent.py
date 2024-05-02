# AUTO GENERATED FILE - DO NOT EDIT

from dash.development.base_component import Component, _explicitize_args


class Upload_ReactComponent(Component):
    """An Upload_ReactComponent component.
The Upload component

Keyword arguments:

- id (string; default 'default-dash-uploader-id'):
    User supplied id of this component.

- cancelButton (boolean; default True):
    Whether or not to have a cancel button.

- chunkSize (number; default 1024 * 1024):
    Size of file chunks to send to server.

- className (string; default 'dash-uploader-default'):
    Class to add to the upload component by default.

- completeStyle (dict; optional):
    Style when upload is completed (upload finished).

- completedClass (string; default 'dash-uploader-completed'):
    Class to add to the upload component when it is complete.

- completedMessage (string; default 'Complete! '):
    Message to display when upload completed.

- dashAppCallbackBump (number; default 0):
    A prop that is monitored by the dash app   Wheneven the value of
    this prop is changed,   all the props are sent to the dash
    application.     This is used in the dash callbacks.

- defaultStyle (dict; optional):
    Style attributes to add to the upload component.

- disableDragAndDrop (boolean; default False):
    Whether or not to allow file drag and drop.

- disabled (boolean; optional):
    Whether or not to allow file uploading.

- disabledClass (string; default 'dash-uploader-disabled'):
    Class to add to the upload component when it is disabled.

- disabledMessage (string; optional):
    Message to display when upload disabled.

- disabledStyle (dict; optional):
    Style when upload is disabled.

- filetypes (list of strings; default undefined):
    List of allowed file types, e.g. ['jpg', 'png'].

- hoveredClass (string; default 'dash-uploader-hovered'):
    Class to add to the upload component when it is hovered.

- maxFileSize (number; default 1024 * 1024 * 10):
    Maximum size per file in bytes.

- maxFiles (number; default 1):
    Maximum number of files that can be uploaded in one session.

- maxTotalSize (number; optional):
    Maximum total size in bytes.

- pauseButton (boolean; default True):
    Whether or not to have a pause button.

- pausedClass (string; default 'dash-uploader-paused'):
    Class to add to the upload component when it is paused.

- service (string; default '/API/dash-uploader'):
    The service to send the files to.

- simultaneousUploads (number; default 1):
    Number of simultaneous uploads to select.

- startButton (boolean; default True):
    Whether or not to have a start button.

- text (string; default 'Click Here to Select a File'):
    The string to display in the upload component.

- totalFilesCount (number; optional):
    Total number of files to be uploaded.

- totalFilesSize (number; optional):
    Total size of uploaded files to be uploaded (Mb).     Mb =
    1024*1024 bytes.

- upload_id (string; default ''):
    The ID for the upload event (for example, session ID).

- uploadedFileNames (list of strings; optional):
    The names of the files uploaded.

- uploadedFilesSize (number; optional):
    Size of uploaded files (Mb). Mb = 1024*1024 bytes.

- uploadingClass (string; default 'dash-uploader-uploading'):
    Class to add to the upload component when it is uploading.

- uploadingStyle (dict; optional):
    Style when upload is in progress."""
    @_explicitize_args
    def __init__(self, maxFiles=Component.UNDEFINED, maxFileSize=Component.UNDEFINED, maxTotalSize=Component.UNDEFINED, chunkSize=Component.UNDEFINED, simultaneousUploads=Component.UNDEFINED, service=Component.UNDEFINED, className=Component.UNDEFINED, hoveredClass=Component.UNDEFINED, disabledClass=Component.UNDEFINED, pausedClass=Component.UNDEFINED, completedClass=Component.UNDEFINED, uploadingClass=Component.UNDEFINED, defaultStyle=Component.UNDEFINED, disabledStyle=Component.UNDEFINED, uploadingStyle=Component.UNDEFINED, completeStyle=Component.UNDEFINED, text=Component.UNDEFINED, disabledMessage=Component.UNDEFINED, completedMessage=Component.UNDEFINED, uploadedFileNames=Component.UNDEFINED, filetypes=Component.UNDEFINED, startButton=Component.UNDEFINED, pauseButton=Component.UNDEFINED, cancelButton=Component.UNDEFINED, disabled=Component.UNDEFINED, disableDragAndDrop=Component.UNDEFINED, onUploadErrorCallback=Component.UNDEFINED, id=Component.UNDEFINED, dashAppCallbackBump=Component.UNDEFINED, upload_id=Component.UNDEFINED, totalFilesCount=Component.UNDEFINED, uploadedFilesSize=Component.UNDEFINED, totalFilesSize=Component.UNDEFINED, **kwargs):
        self._prop_names = ['id', 'cancelButton', 'chunkSize', 'className', 'completeStyle', 'completedClass', 'completedMessage', 'dashAppCallbackBump', 'defaultStyle', 'disableDragAndDrop', 'disabled', 'disabledClass', 'disabledMessage', 'disabledStyle', 'filetypes', 'hoveredClass', 'maxFileSize', 'maxFiles', 'maxTotalSize', 'pauseButton', 'pausedClass', 'service', 'simultaneousUploads', 'startButton', 'text', 'totalFilesCount', 'totalFilesSize', 'upload_id', 'uploadedFileNames', 'uploadedFilesSize', 'uploadingClass', 'uploadingStyle']
        self._type = 'Upload_ReactComponent'
        self._namespace = 'dash_uploader'
        self._valid_wildcard_attributes =            []
        self.available_properties = ['id', 'cancelButton', 'chunkSize', 'className', 'completeStyle', 'completedClass', 'completedMessage', 'dashAppCallbackBump', 'defaultStyle', 'disableDragAndDrop', 'disabled', 'disabledClass', 'disabledMessage', 'disabledStyle', 'filetypes', 'hoveredClass', 'maxFileSize', 'maxFiles', 'maxTotalSize', 'pauseButton', 'pausedClass', 'service', 'simultaneousUploads', 'startButton', 'text', 'totalFilesCount', 'totalFilesSize', 'upload_id', 'uploadedFileNames', 'uploadedFilesSize', 'uploadingClass', 'uploadingStyle']
        self.available_wildcard_properties =            []
        _explicit_args = kwargs.pop('_explicit_args')
        _locals = locals()
        _locals.update(kwargs)  # For wildcard attrs
        args = {k: _locals[k] for k in _explicit_args if k != 'children'}
        for k in []:
            if k not in args:
                raise TypeError(
                    'Required argument `' + k + '` was not specified.')
        super(Upload_ReactComponent, self).__init__(**args)
