mport numpy as np
from . import read_testbase 
import copy
import pandas as pd
from .test import Test

class FMM(object):
    def __init__(self, N, Z, W,test_file):
        self.Z = Z #潜在クラスｚの次元数
        self.W = W #潜在クラスｗの次元数
        self.week = 7 #曜日は７種類
        
        self.df=N
        self.dl=N.values.tolist()#与えられたデータをlistに変換
        self.test_file=test_file
        self.user_list=self.create_list(self.U,N,"user")#self.dlの何行目にuserが現れるかを収納する
        self.item_list=self.create_list(self.I,N,"item")#self.dlの何行目にitemが現れるかを収納する
        self.rating_list=self.create_list(self.R,N,"rating")#self.dlの何行目にratingが現れるかを収納する
        ################################################################################
        #self.user_list[i-1]には顧客iがデータNの何番目に出てくるかを保存している。以下同様
    

        
        #初期化
        # P(z)
        self.Pz =np.random.rand(self.Z)
        # P(w)
        self.Pw =np.random.rand(self.W)
        # P(user|z)
        self.Puser_z = np.random.rand(self.U,self.Z)
        # P(item|w)
        self.Pitem_w = np.random.rand(self.I,self.W)
        # P(rating|zw)
        self.Prating_zw = np.random.rand(self.R,self.Z,self.W)
        

        # 正規化
        self.Pz /= np.sum(self.Pz)
        self.Pw /= np.sum(self.Pw)
        self.Puser_z/=np.sum(self.Puser_z,axis=0)[None,:]
        self.Pitem_w/=np.sum(self.Pitem_w,axis=0)[None,:]
        
        

    def estep(self):#e-stepを行う
        self.tmp=[]
        for x in self.dl: 
            zz=self.Pz*self.Puser_z[x[0]-1]
            ww=self.Pw*self.Pitem_w[x[1]-1]
            self.tmp.append(self.Prating_zw[x[2]-1]*zz[:,None]*ww[None,:])
        for x in range(len(self.tmp)):
            self.tmp[x]/=np.sum(self.tmp[x])
        
    def mstep(self):#m-stepを行う
        goukei=np.zeros(self.Z*self.W).reshape(self.Z,self.W)
        for x in self.tmp:    
            goukei+=x
            
        self.Pz=copy.deepcopy(goukei)
        self.Pw=copy.deepcopy(goukei)
        #P(z),P(w)
        self.Pz=np.sum(self.Pz,axis=1)
        self.Pz/=np.sum(self.Pz)
        self.Pw=np.sum(self.Pw,axis=0)
        self.Pw/=np.sum(self.Pw)

        #P(user|z)
        self.Puser_z=[]
        for x in self.user_list:
            stack=np.zeros(self.Z*self.W).reshape(self.Z,self.W)
            for y in x:
                stack+=self.tmp[y]
            stack=np.sum(stack,axis=1)
            self.Puser_z.append(stack)
        self.Puser_z/=np.sum(self.Puser_z,axis=0)
        #P(item|w)
        self.Pitem_w = []
        for x in self.item_list:
            stack = np.zeros(self.Z*self.W).reshape(self.Z,self.W)
            for y in x:
                stack += self.tmp[y]
            stack=np.sum(stack,axis=0)
            self.Pitem_w.append(stack)
        self.Pitem_w /= np.sum(self.Pitem_w, axis=0)
        #P(rating|z)
        self.Prating_zw = []
        for x in self.rating_list:
            stack = np.zeros(self.Z*self.W).reshape(self.Z,self.W)
            for y in x:
                stack += self.tmp[y]
            self.Prating_zw.append(stack)
        shou=np.zeros(self.Z*self.W).reshape(self.Z,self.W)
        for x in range(self.R):
            shou+=self.Prating_zw[x]
        for x in range(self.R):
            self.Prating_zw[x]/=shou
            
        

    def train(self, k=200, t=1.0e-4):#k:試行回数のmax,t:相対誤差がこれより小さくなれば強制終了
        #対数尤度が収束するまでEステップとMステップを繰り返す
        prev_llh = 100000
        for i in range(k):
            
            self.estep()
            self.mstep()
            '''
            self.Pz[np.isnan(self.Pz)]=0
            self.Pw[np.isnan(self.Pw)]=0
            self.Puser_z[np.isnan(self.Puser_z)]=0
            self.Pitem_w[np.isnan(self.Pitem_w)]=0
            
            for x in range(self.R):
                self.Prating_zw[x][np.isnan(self.Prating_zw[x])] = 0
                '''
            llh = self.llh()

            if abs((llh - prev_llh) / prev_llh) < t:
                break
            prev_llh = llh

    def llh(self):  # 対数尤度の計算関数
        
        Puir=[]#収納list
        for x in self.dl:
            zz=self.Pz*self.Puser_z[x[0]-1]
            ww=self.Pw*self.Pitem_w[x[1]-1]
            stack=self.Prating_zw[x[2]-1]*zz[:,None]*ww[None,:]
            Puir.append(stack)
        
        for x in range(len(Puir)):
            Puir[x] = np.log(Puir[x])#logにする
            Puir[x][np.isinf(Puir[x])]=0#値が小さすぎる場合は0で置き換える。0の根拠はない
        nPuir=np.array(Puir)
        return np.sum(nPuir)#対数尤度LLの値を返す

    def create_list(self,num,N,colname):
        lists=[]
        for x in range(num):
            lists.append([])
        return self.create_containlist(N,colname,lists)
        #ユーザーとアイテムの数をself.U,self.Iで定義する
        df = pd.read_csv("ml-100k/u.info", header=None, delimiter=" ")
        dl=df.values.tolist()
        for x in dl:
            if "users" in x:
                self.U=int(x[0])#ユーザーの数
            elif "items" in x:
                self.I=int(x[0])#アイテムの数

    def create_containlist(self,df,colname,contain):
        i = 0
        dl=df.loc[:,colname].T.values.tolist()
        for x in dl:
            contain[x-1].append(i)
            i += 1
        return contain
    
    def test(self,path):
        a = Test(self.Pz, self.Pw, self.Puser_z, self.Pitem_w,
                 self.Prating_zw, self.df, self.Z, self.W,self.test_file,path)
