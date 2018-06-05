# -*- coding: utf-8 -*-
import wx
import os
from positionfit import positionfit
import threading
import unicodedata

class Window(wx.Frame):

    def __init__(self,*args,**kw):
        super(Window,self).__init__(*args,**kw)
        self.panel_ui = None
        self.TextureModelPath = ""
        self.CTModelPath = ""
        self.checkbox = None
        self.textbox = None
        self.init_ui()


    def init_ui(self):
        self.SetTitle("タイトル")
        self.SetBackgroundColour((180,120,140))
        self.SetPosition((300,300))
        self.SetSize((900,600))

        font = wx.Font(18, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)

        self.panel_ui = wx.Panel(self,-1,pos = (50,50),size = (800,300))
        self.checkbox = wx.CheckBox(self.panel_ui,-1,"大きさを手動で入力するならチェック")
        self.checkbox.Bind(wx.EVT_CHECKBOX,self.state_change_checkbox)
        self.checkbox.SetValue(False)
        self.checkbox.SetFont(font)

        self.textbox = wx.TextCtrl(self.panel_ui,-1)
        self.textbox.Disable()
        self.textbox.SetFont(font)

        self.label1 = wx.StaticText(self.panel_ui, -1, '')
        self.label1.SetFont(font)
        TextureInputBtn = wx.Button(self.panel_ui,-1,"テクスチャ付きモデル入力")
        TextureInputBtn.Bind(wx.EVT_BUTTON,self.InputTextureModel)
        TextureInputBtn.SetFont(font)


        self.label2 = wx.StaticText(self.panel_ui, -1, '')
        self.label2.SetFont(font)
        CTModelInputBtn = wx.Button(self.panel_ui,-1,"CTモデル入力")
        CTModelInputBtn.Bind(wx.EVT_BUTTON,self.InputCTModel)
        CTModelInputBtn.SetFont(font)

        PositionfitBtn = wx.Button(self.panel_ui,-1,"位置合わせ実行")
        PositionfitBtn.Bind(wx.EVT_BUTTON,self.calcPositionfit)
        PositionfitBtn.SetFont(font)


        layout = wx.BoxSizer(wx.VERTICAL)
        layout.Add(self.checkbox,flag=wx.GROW        )
        layout.Add(wx.StaticText(self.panel_ui,-1,""))
        layout.Add(self.textbox ,flag=wx.GROW        )

        layout.Add(wx.StaticText(self.panel_ui,-1,""))
        layout.Add(wx.StaticText(self.panel_ui,-1,""))
        layout.Add(wx.StaticText(self.panel_ui,-1,""))

        layout.Add(self.label1,flag=wx.GROW    )
        layout.Add(TextureInputBtn,flag=wx.GROW)

        layout.Add(wx.StaticText(self.panel_ui,-1,""))
        layout.Add(wx.StaticText(self.panel_ui,-1,""))
        layout.Add(wx.StaticText(self.panel_ui,-1,""))

        layout.Add(self.label2,flag=wx.GROW    )
        layout.Add(CTModelInputBtn,flag=wx.GROW)


        layout.Add(wx.StaticText(self.panel_ui,-1,""))
        layout.Add(wx.StaticText(self.panel_ui,-1,""))
        layout.Add(wx.StaticText(self.panel_ui,-1,""))
        layout.Add(wx.StaticText(self.panel_ui,-1,""))
        layout.Add(wx.StaticText(self.panel_ui,-1,""))


        layout.Add(PositionfitBtn,flag=wx.GROW )

        self.panel_ui.SetSizer(layout)

        self.Show()

    def state_change_checkbox(self,event):
        if self.checkbox.GetValue():
            self.textbox.Enable()
        else:
            self.textbox.Disable()

    def InputTextureModel(self,event):
        dialog = wx.FileDialog(self,"ファイル選択", "", "","*.obj*")

        if dialog.ShowModal()==wx.ID_OK:
            path = unicodedata.normalize("NFC",dialog.GetPath())

            if not(os.path.isfile(path)):
                m_dialog = wx.MessageDialog(self, 'ファイルが存在しません', 'エラー', style=wx.OK)
                m_dialog.ShowModal()
                m_dialog.Destroy()
                return

            self.label1.SetLabel(path)
            self.TextureModelPath = path

        dialog.Destroy()

    def InputCTModel(self,event):
        dialog = wx.FileDialog(self,"ファイル選択", "", "","*.obj*")

        if dialog.ShowModal()==wx.ID_OK:
            path = unicodedata.normalize("NFC",dialog.GetPath())

            if not(os.path.isfile(path)):
                m_dialog = wx.MessageDialog(self, 'ファイルが存在しません', 'エラー', style=wx.OK)
                m_dialog.ShowModal()
                m_dialog.Destroy()
                return

            self.label2.SetLabel(path)
            self.CTModelPath = path

        dialog.Destroy()

    def calcPositionfit(self,event):
        if self.TextureModelPath == "" or self.CTModelPath == "":
            dialog = wx.MessageDialog(self, 'ファイルを入力してください', 'エラー', style=wx.OK)
            dialog.ShowModal()
            dialog.Destroy()
            return

        dialog = wx.FileDialog(self, u'保存するファイルを選択してください',"", "","*.obj*",style=wx.FD_SAVE)
        dialog.ShowModal()

        if dialog.GetPath()=="" or os.path.isdir(dialog.GetPath()):
            return

        if self.checkbox.GetValue():
            inputvarratio = self.textbox.GetValue()

            try:
                float(inputvarratio)
            except ValueError:
                dialog = wx.MessageDialog(self, '数値を入力してください', 'エラー', style=wx.OK)
                dialog.ShowModal()
                return

            check = positionfit(self.CTModelPath,self.TextureModelPath,dialog.GetPath(),True,float(inputvarratio))
        else :
            check = positionfit(self.CTModelPath,self.TextureModelPath,dialog.GetPath(),False)

        if not(check):
            dialog = wx.MessageDialog(self, 'ファイル入力エラー', 'エラー', style=wx.OK)
            dialog.ShowModal()
            dialog.Destroy()


if __name__ == "__main__":
    app = wx.App()
    Window(None)
    app.MainLoop()
