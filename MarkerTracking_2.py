import numpy as np
import cv2
import math

#############
def line_intersect(Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = (int)(Ax1 + uA * (Ax2 - Ax1))
    y = (int)(Ay1 + uA * (Ay2 - Ay1))
 
    return x, y
####
def TwoPointDistance(p1, p2):
    distance = (int)(math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)))
    return distance
####
cameraWidth = 640#720
cameraHeight = 320#480
cap = cv2.VideoCapture(0)
print('camera is opened', cap.isOpened())
cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cameraWidth)  # カメラ画像の横幅を640に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cameraHeight) # カメラ画像の縦幅を320に設定


# 閾値 0-255
threshold_value = 50
L_th = 100
H_th = 300
hough_threshold=50

showImgType = 0
lastShowImgType = 0
showTitle = 'Original image'

measurment = False
initPoints = True
myptuv = [[0 for m in range(2)] for n in range(4)] #4 edge points on monitor of previous loop

while(True):
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        showImgType = (showImgType + 1) % 4
    elif key == ord('r'):
        initPoints = True
    elif key == ord('s'):
        print('start saving')
        #save t, x, y, z to a csv file
        print('saving finished')
        
        
    #TODO (keyの操作)
    #
    #
    #
    #
    
    
    ret, frame = cap.read()
    #グレースケールに変換
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #グレースケールを2値化画像に変換 
    #th, im_th = cv2.threshold(grayFrame, threshold_value, 255, cv2.THRESH_BINARY)
    th, im_th = cv2.threshold(grayFrame, 0, 255, cv2.THRESH_OTSU)
    #print(th)
    threshold_img = cv2.bitwise_not(im_th)
    contours,hierarchy = cv2.findContours(threshold_img, 1, 2)
    
    mask = np.zeros(threshold_img.shape,np.uint8)
    
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        #入力画像のモーメント
        mu = cv2.moments(cnt)
        #モーメントからu,v座標を計算
        if mu["m00"] > 0:
            if cv2.contourArea(cnt) > 400:
                cv2.drawContours(mask,[cnt],0,255,-1)
                
    edge_img = cv2.Canny(mask, L_th, H_th)
     #  Standard Hough Line Transform
    lines = cv2.HoughLines(edge_img, 1, np.pi / 180, hough_threshold, None, 0, 0)
    lnLines = 0
    ptuv = [[0 for m in range(2)] for n in range(4)] #4 edge points on monitor
    nPtuv = [0 for m in range(4)] #number of nearest points of each edge point
    
    # Draw the lines
    if (lines is not None):
        nLines = len(lines)
        pt1 = [[0 for m in range(2)] for n in range(nLines)]
        pt2 = [[0 for m in range(2)] for n in range(nLines)]
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1[i] = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2[i] = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(frame, pt1[i], pt2[i], (0,0,255), 1, cv2.LINE_AA)
            
            for j in range(0, i):
                x1, y1 = pt1[j]
                x2, y2 = pt2[j]
                x3, y3 = pt1[i]
                x4, y4 = pt2[i]
                vector1 = (x2 - x1), (y2 - y1)
                vector2 = (x4 - x3), (y4 - y3)
                unit_vector1 = vector1 / np.linalg.norm(vector1)
                unit_vector2 = vector2 / np.linalg.norm(vector2)
                dot_product = np.dot(unit_vector1, unit_vector2)
                if dot_product > 1:
                    dot_product = 1
                elif dot_product < -1:
                    dot_product = -1
                angle = np.arccos(dot_product) #angle in radian
                if math.degrees(angle) > 60 and math.degrees(angle) < 120:
                    pt = line_intersect(x1, y1, x2, y2, x3, y3, x4, y4)
                    if pt is not None:
                        cv2.circle(frame, pt, 2, (255, 255, 255), -1)
                        _p = 0
                        while _p < 4:
                            pntCnt = 1
                            if nPtuv[_p] > 1:
                                pntCnt = nPtuv[_p]
                            ptCent = ptuv[_p][0] / pntCnt, ptuv[_p][1] / pntCnt
                            pntDis = TwoPointDistance(ptCent, pt)
                            if pntDis < 10 or nPtuv[_p] == 0: # the same group
                                ptuv[_p] = ptuv[_p][0] + pt[0], ptuv[_p][1] + pt[1]
                                nPtuv[_p] = nPtuv[_p] + 1
                                _p = 4
                            _p = _p + 1
                                                        
    #print(nLines)
    
    nEdgePnt = 0
    for p in range(4):
        if nPtuv[p] != 0:
            nEdgePnt = nEdgePnt + 1
            ptuv[p] = (int)(ptuv[p][0] / nPtuv[p]), (int)(ptuv[p][1] / nPtuv[p])
            cv2.circle(frame, ptuv[p], 6, (255, 0, 0), 1)
            
    if nEdgePnt == 4:#4点のエッジを検出したら
        if initPoints == True:#点の順番のリセット
            initPoints = False
            initOrd = [0, 1, 2, 3]
            cw, ch = cameraWidth, cameraHeight
            originalPnt = [[0,ch],[0,0],[cw, 0], [cw, ch]]
            for ed in range(4):
                _ord = -1
                rp = 10000
                for p in range(4):
                    rx, ry = (ptuv[p][0] - originalPnt[ed][0]), (ptuv[p][1] - originalPnt[ed][1])
                    dr = math.sqrt(rx*rx + ry*ry)
                    if dr < rp:
                        rp = dr
                        _ord = p
                initOrd[ed] = _ord
                myptuv[ed] = ptuv[_ord]
            print(initOrd)    
        _ptuv = [[0 for m in range(2)] for n in range(4)] #4 edge points on monitor of previous loop
        #pntDist[0] = TwoPointDistance(ptuv[0], ptuv[1])
        rC = [-1, -1, -1, -1]
        for p in range(4):
            minDist = 10000
            for _p in range(4):
                if _p == rC[0] or _p == rC[1] or _p == rC[2] or _p == rC[3]:
                    continue
                dist_now_prev = TwoPointDistance(myptuv[p], ptuv[_p])
                if dist_now_prev < minDist:
                    minDist = dist_now_prev
                    _ptuv[p] = ptuv[_p]
                    rC[p] = _p
 

        ############　TODO　(x, y, z, roll, pitch, yawの計算)###########
        #U = [0.0, 0.0, 0.0, 0.0]
        #V = [0.0, 0.0, 0.0, 0.0]
        #fx = 
        #fy = 
        #d = 
        for p in range(4):
            myptuv[p] = _ptuv[p]
            a, b = _ptuv[p]
            cv2.putText(frame, ("p" + str(p) + str(myptuv[p])), (a + 20, b + 20),
                cv2.FONT_HERSHEY_PLAIN, 1.0,
                (255, 255, 255), 1, cv2.LINE_AA)
            ###############
            ######U, V#########
            #U[p] = 
            #V[p] = 
        
        ###p0, p1, p2, p3###
        #x = [0, 0, 0, 0]
        #y = [0, 0, 0, 0]
        #z = [0, 0, 0, 0]
        #Roll, Pitch, Yaw = 0, 0, 0
        
        


 
    #TODO (bufferにデータを保存)
    #if measurment == True:
        #t, 各エッジ点（u, v）をあるバッファに保存
        # or save t, x, y, z, roll, pitch, yaw in a buffer when start Measement

        
        

    if lastShowImgType is not showImgType:
        cv2.destroyWindow(showTitle)
    lastShowImgType = showImgType
    
    if showImgType == 0:
        show_img = frame.copy()
        showTitle = 'Original image'
        r_, g_, b_ = 0, 200, 0
       
    elif showImgType == 1:
        show_img = grayFrame.copy()
        showTitle = 'gray image'
        r_, g_, b_ = 255, 0, 0
        
    elif showImgType == 2:
        show_img = threshold_img.copy()
        showTitle = 'binary image'
        r_, g_, b_ = 0, 0, 255
    
    elif showImgType == 3:
        show_img = edge_img.copy()
        showTitle = 'edge image'
        r_, g_, b_ = 0, 0, 255
    
    cv2.imshow(showTitle, show_img)
    
     
cap.release()
cv2.destroyAllWindows()