import wx
import dog_classifier
from flask import Flask

# API URL


class PhotoCtrl(wx.App):
    def __init__(self, redirect=False, filename=None):
        wx.App.__init__(self, redirect, filename)
        self.frame = wx.Frame(None, title='Dog Breeds Classifier')
        self.panel = wx.Panel(self.frame)

        self.frame.statusbar = self.frame.CreateStatusBar(1)
        self.frame.statusbar.SetStatusText('No picture selected.')

        self.PhotoMaxSize = 240
        wx.StaticText(self.panel, label="_____________TOP 5 PREDICTIONS_____________",
                      pos=(300, 25))
        instructions = 'Browse for a DOG image'
        img = wx.Image(240, 240)

        self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY,
                                         wx.Bitmap(img))

        instructLbl = wx.StaticText(self.panel, label=instructions)
        self.photoTxt = wx.TextCtrl(self.panel, size=(200, -1))
        self.browseBtn = wx.Button(self.panel, label='Browse')
        self.resetBtn = wx.Button(self.panel, label='Reset')
        self.resetBtn.Bind(wx.EVT_BUTTON, self.onReset)
        self.browseBtn.Bind(wx.EVT_BUTTON, self.onBrowse)

        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.mainSizer.Add(wx.StaticLine(self.panel, wx.ID_ANY),
                           0, wx.ALL | wx.EXPAND, 5)
        self.mainSizer.Add(instructLbl, 0, wx.ALL, 5)
        self.mainSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        self.sizer.Add(self.photoTxt, 0, wx.ALL, 5)
        self.sizer.Add(self.browseBtn, 0, wx.ALL, 5)
        self.sizer.Add(self.resetBtn, 0, wx.ALL, 5)
        self.mainSizer.Add(self.sizer, 0, wx.ALL, 5)

        self.panel.SetSizer(self.mainSizer)
        self.mainSizer.Fit(self.frame)

        self.panel.Layout()
        self.frame.Show()

    def onReset(self, event):
        self.frame.Close()
        self = PhotoCtrl()
        self.frame.SetSize(600, 400)
        self.frame.SetPosition((400, 150))
        self.panel.Show()
        self.MainLoop()

    def onBrowse(self, event):
        """
        Browse for file
        """
        wildcard = "JPEG files (*.jpg)|*.jpg|(*.jpeg)|*.jpeg|(*.png)|*.png"
        dialog = wx.FileDialog(None, "Choose a file",
                               wildcard=wildcard,
                               style=wx.FC_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.photoTxt.SetValue(dialog.GetPath())
        dialog.Destroy()
        self.onView()
        self.browseBtn.Disable()

    def onView(self):
        filepath = self.photoTxt.GetValue()
        img = wx.Image(filepath, wx.BITMAP_TYPE_ANY)
        self.frame.statusbar.PushStatusText('Working on the image. This could take as long as 30 seconds. Please wait...')
        breeds, probability = dog_classifier.predict(filepath)
        height = 50
        self.frame.statusbar.PushStatusText('Complete!')
        for x in range(len(breeds)):
            wx.StaticText(self.panel, label=str(breeds[x]).replace("_", " ").title(), pos=(300, height))
            wx.StaticText(self.panel, label=str(round(probability[x]*100, 2))+"%", pos=(500, height))
            height += 25

        # scale the image, preserving the aspect ratio
        W = img.GetWidth()
        H = img.GetHeight()
        if W > H:
            NewW = self.PhotoMaxSize
            NewH = self.PhotoMaxSize * H / W
        else:
            NewH = self.PhotoMaxSize
            NewW = self.PhotoMaxSize * W / H
        img = img.Scale(NewW, NewH)

        self.imageCtrl.SetBitmap(wx.BitmapFromImage(img))
        self.panel.Refresh()


app = Flask(__name__)

def main():
    app = PhotoCtrl()
    app.frame.SetSize(600, 400)
    app.frame.SetPosition((400, 150))
    app.panel.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()