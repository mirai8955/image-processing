python�ŉ摜�������s���A���S���Y���ł� 

# 2 次元平面空間の測定  
1 台のカメラ使用時の実空間座標の求め方を示す。各変数を以下のように定義する。 
・L(mm):レンズから壁面までの距離  
・(fx,fy):カメラ焦点距離,x 方向.y 方向(dot)  
・Dx,Dy(mm):カメラに映る壁面の x,y 方向の長さ  
・P(x, y, L): ﾀｰｹﾞｯﾄの実空間座標 (mm)  
・B(u’, v’, fx): ﾀｰｹﾞｯﾄの画像面ピクセル座標 (dot)  
・(u, v): ﾀｰｹﾞｯﾄの PC ｳｨﾝﾄﾞｳ座標 (dot)  

次に実空間座標 P の求め方を以下のように示す。 
まず、PC ウィンドウ座標から画像面ピクセル画像へと座標返還を行い、u’, v’座標を求める。  

$$u’ = u - 320$$

$$v’ = -v + 240$$  

また、fx,fy を以下のように求める。 

$$fx＝(320*L)/(Dx/2)・・・①$$  
$$fx＝(320*L)/(Dy/2)・・・②$$

次に、t を画像面ピクセル座標から実空間座標への変換係数と見做すと、画像面ピクセル座  
標から実空間座標への変換の式は  

$$P(x, y, z) = t B(u’, v’, fx,y)$$

と考えられることができ、x の最大値= Dx/2 について、u’は 320 dot であるから、tx= 
(Dx/2)/320 と考えられる。  
ここで①より、(Dx/2)/320 = L/f と表せられる。したがって、  
実空間座標 P(x, y, z)は  

$$x = tx u’ = (L/fx)u’…③$$  
$$y = ty v’ = (L/fy)v’…④$$  
$$z = L…⑤$$  
と求められることができる。  