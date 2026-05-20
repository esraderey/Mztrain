#!/usr/bin/env python3
"""
ZCodeBERT - Entrenamiento especializado en videojuegos HTML5.

Carga el checkpoint existente y continua entrenamiento con datos
de juegos HTML5 (Canvas, CSS, JavaScript) durante 20 minutos.
"""

import sys
import time
import random
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "src")

from mztrain import (
    ZTrainConfig,
    RankSchedule,
    ZCompressedAdam,
    ZRankScheduler,
    ZGradientCompressor,
    ZActivationCheckpoint,
)
from mztrain.data import CodeTokenizer, CodeDataset, MLMDataset
from mztrain.models import ZCodeBERTConfig, ZCodeBERTForMLM


# =============================================================================
# Dataset de videojuegos HTML5
# =============================================================================

HTML_GAME_SNIPPETS = [
    # ---- SNAKE GAME ----
    """<!DOCTYPE html><html><head><title>Snake</title><style>
canvas{border:2px solid #333;background:#000;display:block;margin:20px auto}
body{background:#1a1a2e;text-align:center;color:#fff;font-family:Arial}
h1{color:#0ff}#score{font-size:24px;color:#0f0}
</style></head><body><h1>Snake Game</h1><p id="score">Score: 0</p>
<canvas id="game" width="400" height="400"></canvas><script>
const canvas=document.getElementById('game');const ctx=canvas.getContext('2d');
const grid=20;let snake=[{x:200,y:200}];let food={x:0,y:0};
let dx=grid;let dy=0;let score=0;let speed=100;
function placeFood(){food.x=Math.floor(Math.random()*20)*grid;
food.y=Math.floor(Math.random()*20)*grid;}placeFood();
document.addEventListener('keydown',e=>{
if(e.key==='ArrowUp'&&dy===0){dx=0;dy=-grid;}
if(e.key==='ArrowDown'&&dy===0){dx=0;dy=grid;}
if(e.key==='ArrowLeft'&&dx===0){dx=-grid;dy=0;}
if(e.key==='ArrowRight'&&dx===0){dx=grid;dy=0;}});
function gameLoop(){const head={x:snake[0].x+dx,y:snake[0].y+dy};
if(head.x<0||head.x>=400||head.y<0||head.y>=400){alert('Game Over! Score: '+score);
snake=[{x:200,y:200}];dx=grid;dy=0;score=0;document.getElementById('score').textContent='Score: '+score;placeFood();return;}
for(let i=0;i<snake.length;i++){if(head.x===snake[i].x&&head.y===snake[i].y){alert('Game Over!');
snake=[{x:200,y:200}];dx=grid;dy=0;score=0;placeFood();return;}}
snake.unshift(head);if(head.x===food.x&&head.y===food.y){score+=10;
document.getElementById('score').textContent='Score: '+score;placeFood();}else{snake.pop();}
ctx.fillStyle='#000';ctx.fillRect(0,0,400,400);
ctx.fillStyle='#0f0';snake.forEach(s=>{ctx.fillRect(s.x,s.y,grid-2,grid-2);});
ctx.fillStyle='#f00';ctx.fillRect(food.x,food.y,grid-2,grid-2);}
setInterval(gameLoop,speed);</script></body></html>""",

    # ---- PONG GAME ----
    """<!DOCTYPE html><html><head><title>Pong</title><style>
canvas{border:2px solid #0ff;background:#111;display:block;margin:20px auto}
body{background:#0a0a23;text-align:center;color:#fff;font-family:monospace}
</style></head><body><h1>Pong</h1><canvas id="pong" width="600" height="400"></canvas>
<script>const canvas=document.getElementById('pong');const ctx=canvas.getContext('2d');
const paddleH=80;const paddleW=10;let leftY=160;let rightY=160;
let ballX=300;let ballY=200;let ballDX=4;let ballDY=3;
let scoreL=0;let scoreR=0;let keys={};
document.addEventListener('keydown',e=>keys[e.key]=true);
document.addEventListener('keyup',e=>keys[e.key]=false);
function update(){if(keys['w']&&leftY>0)leftY-=5;
if(keys['s']&&leftY<320)leftY+=5;
if(keys['ArrowUp']&&rightY>0)rightY-=5;
if(keys['ArrowDown']&&rightY<320)rightY+=5;
ballX+=ballDX;ballY+=ballDY;
if(ballY<=0||ballY>=400)ballDY=-ballDY;
if(ballX<=paddleW&&ballY>=leftY&&ballY<=leftY+paddleH){ballDX=-ballDX;ballDX*=1.05;}
if(ballX>=600-paddleW&&ballY>=rightY&&ballY<=rightY+paddleH){ballDX=-ballDX;ballDX*=1.05;}
if(ballX<0){scoreR++;reset();}if(ballX>600){scoreL++;reset();}}
function reset(){ballX=300;ballY=200;ballDX=4*(Math.random()>0.5?1:-1);ballDY=3*(Math.random()>0.5?1:-1);}
function draw(){ctx.fillStyle='#111';ctx.fillRect(0,0,600,400);
ctx.fillStyle='#0ff';ctx.fillRect(0,leftY,paddleW,paddleH);
ctx.fillRect(590,rightY,paddleW,paddleH);
ctx.beginPath();ctx.arc(ballX,ballY,8,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
ctx.font='32px monospace';ctx.fillText(scoreL,200,50);ctx.fillText(scoreR,380,50);
ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(300,0);ctx.lineTo(300,400);
ctx.strokeStyle='#333';ctx.stroke();}
function loop(){update();draw();requestAnimationFrame(loop);}loop();
</script></body></html>""",

    # ---- BREAKOUT GAME ----
    """<!DOCTYPE html><html><head><title>Breakout</title><style>
canvas{border:2px solid #f0f;background:#000;display:block;margin:20px auto}
body{background:#1a0a2e;text-align:center;color:#fff;font-family:Arial}
</style></head><body><h1>Breakout</h1><canvas id="breakout" width="480" height="320"></canvas>
<script>const c=document.getElementById('breakout');const ctx=c.getContext('2d');
let ballX=240;let ballY=280;let dx=3;let dy=-3;const ballR=6;
let paddleX=190;const paddleW=100;const paddleH=10;
const brickRows=5;const brickCols=8;const brickW=55;const brickH=20;const brickPad=4;
let bricks=[];for(let r=0;r<brickRows;r++){bricks[r]=[];
for(let col=0;col<brickCols;col++){bricks[r][col]={x:col*(brickW+brickPad)+10,
y:r*(brickH+brickPad)+30,alive:true};}}let score=0;let lives=3;
document.addEventListener('mousemove',e=>{let rect=c.getBoundingClientRect();
paddleX=e.clientX-rect.left-paddleW/2;
if(paddleX<0)paddleX=0;if(paddleX>380)paddleX=380;});
const colors=['#f00','#f80','#ff0','#0f0','#00f'];
function gameLoop(){ctx.fillStyle='#000';ctx.fillRect(0,0,480,320);
ballX+=dx;ballY+=dy;if(ballX<ballR||ballX>480-ballR)dx=-dx;
if(ballY<ballR)dy=-dy;if(ballY>310){lives--;if(lives<=0){alert('Game Over! Score:'+score);
lives=3;score=0;resetBricks();}ballX=240;ballY=280;dy=-3;}
if(ballY>=300&&ballX>=paddleX&&ballX<=paddleX+paddleW){dy=-dy;
dx+=((ballX-paddleX-paddleW/2)/paddleW)*4;}
for(let r=0;r<brickRows;r++){for(let col=0;col<brickCols;col++){let b=bricks[r][col];
if(b.alive&&ballX>b.x&&ballX<b.x+brickW&&ballY>b.y&&ballY<b.y+brickH){
dy=-dy;b.alive=false;score+=10;}}}
for(let r=0;r<brickRows;r++){for(let col=0;col<brickCols;col++){let b=bricks[r][col];
if(b.alive){ctx.fillStyle=colors[r];ctx.fillRect(b.x,b.y,brickW,brickH);}}}
ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(ballX,ballY,ballR,0,Math.PI*2);ctx.fill();
ctx.fillStyle='#0ff';ctx.fillRect(paddleX,305,paddleW,paddleH);
ctx.fillStyle='#fff';ctx.font='14px Arial';ctx.fillText('Score:'+score+' Lives:'+lives,10,20);
requestAnimationFrame(gameLoop);}function resetBricks(){for(let r=0;r<brickRows;r++)
for(let c=0;c<brickCols;c++)bricks[r][c].alive=true;}gameLoop();
</script></body></html>""",

    # ---- FLAPPY BIRD ----
    """<!DOCTYPE html><html><head><title>Flappy Bird</title><style>
canvas{border:2px solid #4a4;background:#70c5ce;display:block;margin:20px auto}
body{background:#2d6a4f;text-align:center;color:#fff;font-family:Arial}
</style></head><body><h1>Flappy Bird</h1><canvas id="flappy" width="320" height="480"></canvas>
<script>const c=document.getElementById('flappy');const ctx=c.getContext('2d');
let birdY=240;let birdVel=0;const gravity=0.4;const jump=-7;const birdX=60;const birdSize=20;
let pipes=[];let score=0;let frame=0;let gameOver=false;
function addPipe(){let gap=120;let topH=Math.random()*200+50;
pipes.push({x:320,top:topH,bottom:topH+gap});}addPipe();
document.addEventListener('keydown',e=>{if(e.code==='Space'){
if(gameOver){birdY=240;birdVel=0;pipes=[];score=0;gameOver=false;addPipe();}
else{birdVel=jump;}}});
document.addEventListener('click',()=>{if(!gameOver)birdVel=jump;});
function update(){if(gameOver)return;birdVel+=gravity;birdY+=birdVel;
if(birdY>460||birdY<0){gameOver=true;return;}
frame++;if(frame%90===0)addPipe();
for(let i=pipes.length-1;i>=0;i--){pipes[i].x-=2;
if(pipes[i].x<-50){pipes.splice(i,1);score++;continue;}
if(birdX+birdSize>pipes[i].x&&birdX<pipes[i].x+40){
if(birdY<pipes[i].top||birdY+birdSize>pipes[i].bottom){gameOver=true;}}}}
function draw(){ctx.fillStyle='#70c5ce';ctx.fillRect(0,0,320,480);
ctx.fillStyle='#2ecc71';pipes.forEach(p=>{ctx.fillRect(p.x,0,40,p.top);
ctx.fillRect(p.x,p.bottom,40,480-p.bottom);});
ctx.fillStyle='#f1c40f';ctx.fillRect(birdX,birdY,birdSize,birdSize);
ctx.fillStyle='#fff';ctx.font='24px Arial';ctx.fillText(score,150,40);
if(gameOver){ctx.fillStyle='rgba(0,0,0,0.5)';ctx.fillRect(0,0,320,480);
ctx.fillStyle='#fff';ctx.font='30px Arial';ctx.fillText('GAME OVER',80,240);
ctx.font='16px Arial';ctx.fillText('Press Space to restart',75,280);}}
function loop(){update();draw();requestAnimationFrame(loop);}loop();
</script></body></html>""",

    # ---- SPACE INVADERS ----
    """<!DOCTYPE html><html><head><title>Space Invaders</title><style>
canvas{border:2px solid #0f0;background:#000;display:block;margin:20px auto}
body{background:#0a0a0a;text-align:center;color:#0f0;font-family:monospace}
</style></head><body><h1>Space Invaders</h1><canvas id="si" width="400" height="500"></canvas>
<script>const c=document.getElementById('si');const ctx=c.getContext('2d');
let player={x:180,y:460,w:40,h:20};let bullets=[];let enemies=[];let score=0;
let keys={};let enemyDir=1;let enemySpeed=1;
for(let row=0;row<4;row++){for(let col=0;col<8;col++){
enemies.push({x:col*45+20,y:row*35+40,w:30,h:20,alive:true});}}
document.addEventListener('keydown',e=>keys[e.key]=true);
document.addEventListener('keyup',e=>keys[e.key]=false);
function update(){if(keys['ArrowLeft']&&player.x>0)player.x-=5;
if(keys['ArrowRight']&&player.x<360)player.x+=5;
if(keys[' ']&&bullets.length<5){bullets.push({x:player.x+18,y:player.y-5});keys[' ']=false;}
bullets.forEach(b=>b.y-=6);bullets=bullets.filter(b=>b.y>0);
let moveDown=false;enemies.forEach(e=>{if(e.alive){e.x+=enemyDir*enemySpeed;
if(e.x>370||e.x<0)moveDown=true;}});
if(moveDown){enemyDir*=-1;enemies.forEach(e=>{if(e.alive)e.y+=20;});}
for(let i=bullets.length-1;i>=0;i--){for(let j=0;j<enemies.length;j++){
let e=enemies[j];if(e.alive&&bullets[i]&&bullets[i].x>e.x&&bullets[i].x<e.x+e.w&&
bullets[i].y>e.y&&bullets[i].y<e.y+e.h){e.alive=false;bullets.splice(i,1);score+=100;break;}}}
enemies.forEach(e=>{if(e.alive&&e.y>player.y){alert('Game Over! Score:'+score);resetGame();}});}
function resetGame(){score=0;enemies.forEach((e,i)=>{e.alive=true;
e.x=(i%8)*45+20;e.y=Math.floor(i/8)*35+40;});player.x=180;bullets=[];}
function draw(){ctx.fillStyle='#000';ctx.fillRect(0,0,400,500);
ctx.fillStyle='#0f0';ctx.fillRect(player.x,player.y,player.w,player.h);
ctx.fillStyle='#ff0';bullets.forEach(b=>ctx.fillRect(b.x,b.y,4,10));
ctx.fillStyle='#f00';enemies.forEach(e=>{if(e.alive)ctx.fillRect(e.x,e.y,e.w,e.h);});
ctx.fillStyle='#0f0';ctx.font='16px monospace';ctx.fillText('SCORE: '+score,10,25);}
function loop(){update();draw();requestAnimationFrame(loop);}loop();
</script></body></html>""",

    # ---- TETRIS ----
    """<!DOCTYPE html><html><head><title>Tetris</title><style>
canvas{border:2px solid #fff;background:#111;display:block;margin:20px auto}
body{background:#1a1a2e;text-align:center;color:#fff;font-family:Arial}
</style></head><body><h1>Tetris</h1><p id="sc">Score: 0</p>
<canvas id="tetris" width="200" height="400"></canvas><script>
const c=document.getElementById('tetris');const ctx=c.getContext('2d');
const COLS=10;const ROWS=20;const SZ=20;
const SHAPES=[[[1,1,1,1]],[[1,1],[1,1]],[[0,1,0],[1,1,1]],
[[1,0,0],[1,1,1]],[[0,0,1],[1,1,1]],[[1,1,0],[0,1,1]],[[0,1,1],[1,1,0]]];
const COLORS=['#0ff','#ff0','#a0f','#00f','#f80','#0f0','#f00'];
let board=Array.from({length:ROWS},()=>Array(COLS).fill(0));
let current;let curX;let curY;let curColor;let score=0;let dropTime=0;
function newPiece(){let idx=Math.floor(Math.random()*SHAPES.length);
current=SHAPES[idx];curColor=COLORS[idx];curX=3;curY=0;
if(collides(curX,curY,current)){alert('Game Over! Score:'+score);
board=Array.from({length:ROWS},()=>Array(COLS).fill(0));score=0;}}
function collides(px,py,piece){for(let r=0;r<piece.length;r++)
for(let c=0;c<piece[r].length;c++)if(piece[r][c]&&
(py+r>=ROWS||px+c<0||px+c>=COLS||board[py+r][px+c]))return true;return false;}
function merge(){for(let r=0;r<current.length;r++)for(let c=0;c<current[r].length;c++)
if(current[r][c])board[curY+r][curX+c]=curColor;}
function clearLines(){for(let r=ROWS-1;r>=0;r--){if(board[r].every(c=>c)){
board.splice(r,1);board.unshift(Array(COLS).fill(0));score+=100;r++;}}}
function rotate(piece){let h=piece.length;let w=piece[0].length;
let rotated=Array.from({length:w},()=>Array(h).fill(0));
for(let r=0;r<h;r++)for(let c=0;c<w;c++)rotated[c][h-1-r]=piece[r][c];return rotated;}
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'&&!collides(curX-1,curY,current))curX--;
if(e.key==='ArrowRight'&&!collides(curX+1,curY,current))curX++;
if(e.key==='ArrowDown'&&!collides(curX,curY+1,current))curY++;
if(e.key==='ArrowUp'){let rot=rotate(current);if(!collides(curX,curY,rot))current=rot;}});
function draw(){ctx.fillStyle='#111';ctx.fillRect(0,0,200,400);
for(let r=0;r<ROWS;r++)for(let c=0;c<COLS;c++)if(board[r][c]){
ctx.fillStyle=board[r][c];ctx.fillRect(c*SZ,r*SZ,SZ-1,SZ-1);}
ctx.fillStyle=curColor;for(let r=0;r<current.length;r++)for(let c=0;c<current[r].length;c++)
if(current[r][c])ctx.fillRect((curX+c)*SZ,(curY+r)*SZ,SZ-1,SZ-1);}
newPiece();function loop(){dropTime++;if(dropTime>30){dropTime=0;
if(!collides(curX,curY+1,current)){curY++;}else{merge();clearLines();newPiece();}}
draw();document.getElementById('sc').textContent='Score: '+score;
requestAnimationFrame(loop);}loop();</script></body></html>""",

    # ---- ASTEROIDS ----
    """<!DOCTYPE html><html><head><title>Asteroids</title><style>
canvas{border:1px solid #555;background:#000;display:block;margin:20px auto}
body{background:#0a0a1a;text-align:center;color:#fff;font-family:monospace}
</style></head><body><h1>Asteroids</h1><canvas id="ast" width="600" height="400"></canvas>
<script>const c=document.getElementById('ast');const ctx=c.getContext('2d');
let ship={x:300,y:200,angle:0,vx:0,vy:0};let bullets=[];let asteroids=[];let score=0;
let keys={};for(let i=0;i<5;i++){asteroids.push({x:Math.random()*600,
y:Math.random()*400,vx:(Math.random()-0.5)*2,vy:(Math.random()-0.5)*2,r:30});}
document.addEventListener('keydown',e=>keys[e.key]=true);
document.addEventListener('keyup',e=>keys[e.key]=false);
function update(){if(keys['ArrowLeft'])ship.angle-=0.05;
if(keys['ArrowRight'])ship.angle+=0.05;
if(keys['ArrowUp']){ship.vx+=Math.cos(ship.angle)*0.1;ship.vy+=Math.sin(ship.angle)*0.1;}
ship.x+=ship.vx;ship.y+=ship.vy;
ship.x=(ship.x+600)%600;ship.y=(ship.y+400)%400;
ship.vx*=0.99;ship.vy*=0.99;
if(keys[' ']&&bullets.length<10){bullets.push({x:ship.x,y:ship.y,
vx:Math.cos(ship.angle)*5,vy:Math.sin(ship.angle)*5,life:60});keys[' ']=false;}
bullets.forEach(b=>{b.x+=b.vx;b.y+=b.vy;b.life--;});
bullets=bullets.filter(b=>b.life>0);
asteroids.forEach(a=>{a.x+=a.vx;a.y+=a.vy;a.x=(a.x+600)%600;a.y=(a.y+400)%400;});
for(let i=bullets.length-1;i>=0;i--){for(let j=asteroids.length-1;j>=0;j--){
let dx=bullets[i].x-asteroids[j].x;let dy=bullets[i].y-asteroids[j].y;
if(Math.sqrt(dx*dx+dy*dy)<asteroids[j].r){
if(asteroids[j].r>15){asteroids.push({x:asteroids[j].x,y:asteroids[j].y,
vx:(Math.random()-0.5)*3,vy:(Math.random()-0.5)*3,r:asteroids[j].r/2});
asteroids.push({x:asteroids[j].x,y:asteroids[j].y,
vx:(Math.random()-0.5)*3,vy:(Math.random()-0.5)*3,r:asteroids[j].r/2});}
asteroids.splice(j,1);bullets.splice(i,1);score+=50;break;}}}}
function draw(){ctx.fillStyle='#000';ctx.fillRect(0,0,600,400);
ctx.save();ctx.translate(ship.x,ship.y);ctx.rotate(ship.angle);
ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(15,0);ctx.lineTo(-10,-8);
ctx.lineTo(-10,8);ctx.closePath();ctx.stroke();ctx.restore();
ctx.fillStyle='#ff0';bullets.forEach(b=>{ctx.beginPath();ctx.arc(b.x,b.y,2,0,Math.PI*2);ctx.fill();});
ctx.strokeStyle='#888';asteroids.forEach(a=>{ctx.beginPath();ctx.arc(a.x,a.y,a.r,0,Math.PI*2);ctx.stroke();});
ctx.fillStyle='#fff';ctx.font='16px monospace';ctx.fillText('Score: '+score,10,25);}
function loop(){update();draw();requestAnimationFrame(loop);}loop();
</script></body></html>""",

    # ---- PLATFORMER ----
    """<!DOCTYPE html><html><head><title>Platformer</title><style>
canvas{border:2px solid #4a4;background:#87ceeb;display:block;margin:20px auto}
body{background:#2d5a27;text-align:center;color:#fff;font-family:Arial}
</style></head><body><h1>Platformer</h1><canvas id="plat" width="600" height="400"></canvas>
<script>const c=document.getElementById('plat');const ctx=c.getContext('2d');
let player={x:50,y:300,w:30,h:30,vx:0,vy:0,onGround:false,jumps:0};
let platforms=[{x:0,y:370,w:600,h:30},{x:100,y:300,w:100,h:15},
{x:250,y:250,w:100,h:15},{x:400,y:200,w:120,h:15},{x:150,y:150,w:80,h:15},
{x:350,y:100,w:100,h:15},{x:50,y:50,w:80,h:15}];
let coins=[];let score=0;let keys={};
platforms.forEach(p=>{if(p.y<370)coins.push({x:p.x+p.w/2-8,y:p.y-25,w:16,h:16,taken:false});});
document.addEventListener('keydown',e=>keys[e.key]=true);
document.addEventListener('keyup',e=>keys[e.key]=false);
function update(){if(keys['ArrowLeft'])player.vx=-4;
else if(keys['ArrowRight'])player.vx=4;
else player.vx*=0.8;
if(keys['ArrowUp']&&player.jumps<2){player.vy=-10;player.jumps++;keys['ArrowUp']=false;}
player.vy+=0.5;player.x+=player.vx;player.y+=player.vy;player.onGround=false;
platforms.forEach(p=>{if(player.x+player.w>p.x&&player.x<p.x+p.w&&
player.y+player.h>p.y&&player.y+player.h<p.y+p.h+10&&player.vy>0){
player.y=p.y-player.h;player.vy=0;player.onGround=true;player.jumps=0;}});
if(player.x<0)player.x=0;if(player.x>570)player.x=570;
if(player.y>400){player.x=50;player.y=300;player.vy=0;}
coins.forEach(coin=>{if(!coin.taken&&player.x+player.w>coin.x&&player.x<coin.x+coin.w&&
player.y+player.h>coin.y&&player.y<coin.y+coin.h){coin.taken=true;score+=50;}});}
function draw(){ctx.fillStyle='#87ceeb';ctx.fillRect(0,0,600,400);
ctx.fillStyle='#4a2';platforms.forEach(p=>ctx.fillRect(p.x,p.y,p.w,p.h));
ctx.fillStyle='#f00';ctx.fillRect(player.x,player.y,player.w,player.h);
ctx.fillStyle='#ff0';coins.forEach(coin=>{if(!coin.taken){
ctx.beginPath();ctx.arc(coin.x+8,coin.y+8,8,0,Math.PI*2);ctx.fill();}});
ctx.fillStyle='#000';ctx.font='18px Arial';ctx.fillText('Score: '+score,10,25);}
function loop(){update();draw();requestAnimationFrame(loop);}loop();
</script></body></html>""",

    # ---- MEMORY CARD GAME ----
    """<!DOCTYPE html><html><head><title>Memory</title><style>
body{background:#1a1a2e;text-align:center;color:#fff;font-family:Arial}
.board{display:grid;grid-template-columns:repeat(4,80px);gap:10px;justify-content:center;margin:20px auto}
.card{width:80px;height:80px;background:#16213e;border:2px solid #0f3460;border-radius:8px;
cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:32px;
transition:transform 0.3s;user-select:none}.card:hover{border-color:#e94560}
.card.flipped{background:#0f3460;transform:rotateY(180deg)}.card.matched{background:#2ecc71;border-color:#27ae60}
#info{font-size:20px;margin:10px}
</style></head><body><h1>Memory Game</h1><p id="info">Moves: 0 | Pairs: 0/8</p>
<div class="board" id="board"></div><script>
const emojis=['A','B','C','D','E','F','G','H'];
let cards=[...emojis,...emojis];let flipped=[];let matched=0;let moves=0;let locked=false;
function shuffle(a){for(let i=a.length-1;i>0;i--){let j=Math.floor(Math.random()*(i+1));
[a[i],a[j]]=[a[j],a[i]];}return a;}shuffle(cards);
const board=document.getElementById('board');
cards.forEach((sym,i)=>{let card=document.createElement('div');card.className='card';
card.dataset.idx=i;card.dataset.sym=sym;card.textContent='?';
card.addEventListener('click',()=>flipCard(card));board.appendChild(card);});
function flipCard(card){if(locked||card.classList.contains('flipped')||
card.classList.contains('matched'))return;
card.classList.add('flipped');card.textContent=card.dataset.sym;flipped.push(card);
if(flipped.length===2){moves++;locked=true;
if(flipped[0].dataset.sym===flipped[1].dataset.sym){
flipped[0].classList.add('matched');flipped[1].classList.add('matched');
matched++;flipped=[];locked=false;
if(matched===8){setTimeout(()=>alert('You won in '+moves+' moves!'),300);}}
else{setTimeout(()=>{flipped[0].classList.remove('flipped');flipped[0].textContent='?';
flipped[1].classList.remove('flipped');flipped[1].textContent='?';
flipped=[];locked=false;},800);}
document.getElementById('info').textContent='Moves: '+moves+' | Pairs: '+matched+'/8';}}
</script></body></html>""",

    # ---- SHOOTER TOP-DOWN ----
    """<!DOCTYPE html><html><head><title>Shooter</title><style>
canvas{border:2px solid #f00;background:#111;display:block;margin:20px auto}
body{background:#1a0505;text-align:center;color:#f44;font-family:monospace}
</style></head><body><h1>Top-Down Shooter</h1><canvas id="shoot" width="500" height="500"></canvas>
<script>const c=document.getElementById('shoot');const ctx=c.getContext('2d');
let player={x:250,y:250,angle:0,hp:100};let bullets=[];let enemies=[];
let score=0;let keys={};let mouseX=250;let mouseY=250;let wave=1;
document.addEventListener('keydown',e=>keys[e.key]=true);
document.addEventListener('keyup',e=>keys[e.key]=false);
c.addEventListener('mousemove',e=>{let r=c.getBoundingClientRect();
mouseX=e.clientX-r.left;mouseY=e.clientY-r.top;});
c.addEventListener('click',()=>{let angle=Math.atan2(mouseY-player.y,mouseX-player.x);
bullets.push({x:player.x,y:player.y,vx:Math.cos(angle)*8,vy:Math.sin(angle)*8});});
function spawnWave(){for(let i=0;i<wave*3;i++){let side=Math.floor(Math.random()*4);
let ex,ey;if(side===0){ex=Math.random()*500;ey=-20;}
else if(side===1){ex=520;ey=Math.random()*500;}
else if(side===2){ex=Math.random()*500;ey=520;}
else{ex=-20;ey=Math.random()*500;}
enemies.push({x:ex,y:ey,hp:2,speed:1+wave*0.2});}}spawnWave();
function update(){let speed=3;if(keys['w'])player.y-=speed;if(keys['s'])player.y+=speed;
if(keys['a'])player.x-=speed;if(keys['d'])player.x+=speed;
player.x=Math.max(10,Math.min(490,player.x));player.y=Math.max(10,Math.min(490,player.y));
player.angle=Math.atan2(mouseY-player.y,mouseX-player.x);
bullets.forEach(b=>{b.x+=b.vx;b.y+=b.vy;});
bullets=bullets.filter(b=>b.x>0&&b.x<500&&b.y>0&&b.y<500);
enemies.forEach(e=>{let dx=player.x-e.x;let dy=player.y-e.y;
let d=Math.sqrt(dx*dx+dy*dy);if(d>0){e.x+=dx/d*e.speed;e.y+=dy/d*e.speed;}
if(d<20){player.hp-=1;}});
for(let i=bullets.length-1;i>=0;i--){for(let j=enemies.length-1;j>=0;j--){
let dx=bullets[i].x-enemies[j].x;let dy=bullets[i].y-enemies[j].y;
if(Math.sqrt(dx*dx+dy*dy)<15){enemies[j].hp--;bullets.splice(i,1);
if(enemies[j].hp<=0){enemies.splice(j,1);score+=100;}break;}}}
if(enemies.length===0){wave++;spawnWave();}
if(player.hp<=0){alert('Game Over! Score: '+score+' Wave: '+wave);
player.hp=100;player.x=250;player.y=250;score=0;wave=1;enemies=[];spawnWave();}}
function draw(){ctx.fillStyle='#111';ctx.fillRect(0,0,500,500);
ctx.save();ctx.translate(player.x,player.y);ctx.rotate(player.angle);
ctx.fillStyle='#0af';ctx.fillRect(-12,-8,24,16);ctx.fillRect(10,-3,10,6);
ctx.restore();ctx.fillStyle='#ff0';bullets.forEach(b=>{
ctx.beginPath();ctx.arc(b.x,b.y,3,0,Math.PI*2);ctx.fill();});
ctx.fillStyle='#f44';enemies.forEach(e=>{ctx.beginPath();ctx.arc(e.x,e.y,10,0,Math.PI*2);ctx.fill();});
ctx.fillStyle='#fff';ctx.font='14px monospace';
ctx.fillText('HP:'+player.hp+' Score:'+score+' Wave:'+wave,10,20);
ctx.fillStyle='#0f0';ctx.fillRect(10,30,player.hp*2,8);}
function loop(){update();draw();requestAnimationFrame(loop);}loop();
</script></body></html>""",

    # ---- MAZE GAME ----
    """<!DOCTYPE html><html><head><title>Maze</title><style>
canvas{border:2px solid #fff;background:#222;display:block;margin:20px auto}
body{background:#111;text-align:center;color:#fff;font-family:Arial}
</style></head><body><h1>Maze Runner</h1><canvas id="maze" width="400" height="400"></canvas>
<script>const c=document.getElementById('maze');const ctx=c.getContext('2d');const S=40;
const maze=[[1,1,1,1,1,1,1,1,1,1],[1,0,0,0,1,0,0,0,0,1],[1,0,1,0,1,0,1,1,0,1],
[1,0,1,0,0,0,0,1,0,1],[1,0,1,1,1,1,0,1,0,1],[1,0,0,0,0,1,0,0,0,1],
[1,1,1,0,1,1,1,1,0,1],[1,0,0,0,0,0,0,0,0,1],[1,0,1,1,1,1,1,1,0,1],[1,1,1,1,1,1,1,1,1,1]];
let px=1;let py=1;let ex=8;let ey=8;let moves=0;let startTime=Date.now();
document.addEventListener('keydown',e=>{let nx=px;let ny=py;
if(e.key==='ArrowUp')ny--;if(e.key==='ArrowDown')ny++;
if(e.key==='ArrowLeft')nx--;if(e.key==='ArrowRight')nx++;
if(nx>=0&&nx<10&&ny>=0&&ny<10&&maze[ny][nx]===0){px=nx;py=ny;moves++;
if(px===ex&&py===ey){let time=((Date.now()-startTime)/1000).toFixed(1);
alert('You escaped! Moves: '+moves+' Time: '+time+'s');px=1;py=1;moves=0;startTime=Date.now();}}});
function draw(){ctx.fillStyle='#222';ctx.fillRect(0,0,400,400);
for(let y=0;y<10;y++)for(let x=0;x<10;x++){
if(maze[y][x]===1){ctx.fillStyle='#555';ctx.fillRect(x*S,y*S,S,S);}
else{ctx.fillStyle='#1a1a2e';ctx.fillRect(x*S,y*S,S,S);}}
ctx.fillStyle='#0f0';ctx.beginPath();ctx.arc(ex*S+S/2,ey*S+S/2,12,0,Math.PI*2);ctx.fill();
ctx.fillStyle='#f00';ctx.beginPath();ctx.arc(px*S+S/2,py*S+S/2,12,0,Math.PI*2);ctx.fill();
ctx.fillStyle='#fff';ctx.font='14px Arial';ctx.fillText('Moves: '+moves,10,395);
requestAnimationFrame(draw);}draw();</script></body></html>""",

    # ---- COOKIE CLICKER ----
    """<!DOCTYPE html><html><head><title>Cookie Clicker</title><style>
body{background:#2c1810;text-align:center;color:#fff;font-family:Georgia}
#cookie{width:200px;height:200px;border-radius:50%;background:radial-gradient(#d4a574,#8b6914);
border:4px solid #654321;cursor:pointer;margin:20px auto;font-size:60px;line-height:200px;
transition:transform 0.1s;user-select:none}#cookie:active{transform:scale(0.95)}
.shop{display:inline-block;background:#1a0f0a;border:2px solid #654321;padding:10px 20px;
margin:5px;cursor:pointer;border-radius:5px;color:#d4a574}
.shop:hover{background:#2a1f1a}#stats{font-size:24px;color:#d4a574;margin:10px}
</style></head><body><h1>Cookie Clicker</h1><p id="stats">Cookies: 0</p>
<div id="cookie">C</div><p id="cps">Per second: 0</p>
<div><div class="shop" id="b1">Cursor (10)</div><div class="shop" id="b2">Grandma (50)</div>
<div class="shop" id="b3">Farm (200)</div><div class="shop" id="b4">Factory (1000)</div></div>
<script>let cookies=0;let cps=0;let cursors=0;let grandmas=0;let farms=0;let factories=0;
document.getElementById('cookie').addEventListener('click',()=>{
cookies++;updateDisplay();});
document.getElementById('b1').addEventListener('click',()=>{
if(cookies>=10){cookies-=10;cursors++;cps+=0.1;updateDisplay();}});
document.getElementById('b2').addEventListener('click',()=>{
if(cookies>=50){cookies-=50;grandmas++;cps+=1;updateDisplay();}});
document.getElementById('b3').addEventListener('click',()=>{
if(cookies>=200){cookies-=200;farms++;cps+=5;updateDisplay();}});
document.getElementById('b4').addEventListener('click',()=>{
if(cookies>=1000){cookies-=1000;factories++;cps+=20;updateDisplay();}});
function updateDisplay(){document.getElementById('stats').textContent=
'Cookies: '+Math.floor(cookies);document.getElementById('cps').textContent=
'Per second: '+cps.toFixed(1);}
setInterval(()=>{cookies+=cps/10;updateDisplay();},100);
</script></body></html>""",

    # ---- DODGE GAME ----
    """<!DOCTYPE html><html><head><title>Dodge</title><style>
canvas{border:2px solid #f0f;background:#0a0020;display:block;margin:20px auto}
body{background:#050010;text-align:center;color:#f0f;font-family:monospace}
</style></head><body><h1>Dodge!</h1><canvas id="dodge" width="400" height="600"></canvas>
<script>const c=document.getElementById('dodge');const ctx=c.getContext('2d');
let player={x:200,y:550,w:30,h:30};let obstacles=[];let score=0;
let speed=3;let frame=0;let keys={};let gameOver=false;
document.addEventListener('keydown',e=>keys[e.key]=true);
document.addEventListener('keyup',e=>keys[e.key]=false);
function spawn(){let w=20+Math.random()*60;obstacles.push({x:Math.random()*(400-w),y:-20,w:w,h:20});}
function update(){if(gameOver)return;frame++;score++;if(frame%20===0)spawn();
if(frame%500===0)speed+=0.5;
if(keys['ArrowLeft'])player.x-=5;if(keys['ArrowRight'])player.x+=5;
player.x=Math.max(0,Math.min(370,player.x));
obstacles.forEach(o=>o.y+=speed);obstacles=obstacles.filter(o=>o.y<620);
for(let o of obstacles){if(player.x<o.x+o.w&&player.x+player.w>o.x&&
player.y<o.y+o.h&&player.y+player.h>o.y){gameOver=true;
setTimeout(()=>{alert('Score: '+score);score=0;speed=3;obstacles=[];gameOver=false;},100);}}}
function draw(){ctx.fillStyle='#0a0020';ctx.fillRect(0,0,400,600);
let gradient=ctx.createLinearGradient(0,0,0,600);
gradient.addColorStop(0,'#0a0020');gradient.addColorStop(1,'#1a0040');
ctx.fillStyle=gradient;ctx.fillRect(0,0,400,600);
for(let i=0;i<50;i++){ctx.fillStyle='rgba(255,255,255,'+(Math.random()*0.5)+')';
ctx.fillRect(Math.random()*400,Math.random()*600,1,1);}
ctx.fillStyle='#0ff';ctx.fillRect(player.x,player.y,player.w,player.h);
ctx.fillStyle='#f0f';obstacles.forEach(o=>ctx.fillRect(o.x,o.y,o.w,o.h));
ctx.fillStyle='#fff';ctx.font='18px monospace';ctx.fillText('Score: '+score,10,25);}
function loop(){update();draw();requestAnimationFrame(loop);}loop();
</script></body></html>""",

    # ---- RHYTHM GAME ----
    """<!DOCTYPE html><html><head><title>Rhythm</title><style>
canvas{border:2px solid #f80;background:#111;display:block;margin:20px auto}
body{background:#1a0a00;text-align:center;color:#f80;font-family:Arial}
</style></head><body><h1>Rhythm Game</h1><canvas id="rhythm" width="400" height="500"></canvas>
<script>const c=document.getElementById('rhythm');const ctx=c.getContext('2d');
const lanes=[80,160,240,320];const laneKeys=['d','f','j','k'];
const laneColors=['#f00','#0f0','#00f','#ff0'];
let notes=[];let score=0;let combo=0;let maxCombo=0;let frame=0;
let keys={};let hits=[];
document.addEventListener('keydown',e=>{keys[e.key]=true;
let lane=laneKeys.indexOf(e.key);if(lane>=0)checkHit(lane);});
document.addEventListener('keyup',e=>keys[e.key]=false);
function spawnNote(){let lane=Math.floor(Math.random()*4);
notes.push({lane:lane,y:-20,speed:3+Math.random()*2});}
function checkHit(lane){for(let i=notes.length-1;i>=0;i--){
if(notes[i].lane===lane&&notes[i].y>420&&notes[i].y<480){
notes.splice(i,1);score+=100+combo*10;combo++;
if(combo>maxCombo)maxCombo=combo;hits.push({x:lanes[lane],y:450,t:15});return;}}
combo=0;}
function update(){frame++;if(frame%30===0)spawnNote();
notes.forEach(n=>n.y+=n.speed);
for(let i=notes.length-1;i>=0;i--){if(notes[i].y>500){notes.splice(i,1);combo=0;}}
hits=hits.filter(h=>{h.t--;return h.t>0;});}
function draw(){ctx.fillStyle='#111';ctx.fillRect(0,0,400,500);
lanes.forEach((x,i)=>{ctx.strokeStyle='#333';ctx.beginPath();
ctx.moveTo(x,0);ctx.lineTo(x,500);ctx.stroke();
ctx.fillStyle=keys[laneKeys[i]]?'rgba(255,255,255,0.3)':'rgba(255,255,255,0.05)';
ctx.fillRect(x-30,0,60,500);});
ctx.fillStyle='rgba(255,255,255,0.2)';ctx.fillRect(0,440,400,20);
notes.forEach(n=>{ctx.fillStyle=laneColors[n.lane];
ctx.fillRect(lanes[n.lane]-15,n.y,30,20);});
hits.forEach(h=>{ctx.fillStyle='rgba(255,255,255,'+h.t/15+')';
ctx.beginPath();ctx.arc(h.x,h.y,20*(1-h.t/15),0,Math.PI*2);ctx.fill();});
ctx.fillStyle='#fff';ctx.font='16px Arial';
ctx.fillText('Score: '+score+'  Combo: '+combo+'  Max: '+maxCombo,10,25);
ctx.fillStyle='#888';ctx.font='12px Arial';ctx.fillText('D  F  J  K',140,490);}
function loop(){update();draw();requestAnimationFrame(loop);}loop();
</script></body></html>""",

    # ---- TOWER DEFENSE (simplified) ----
    """<!DOCTYPE html><html><head><title>Tower Defense</title><style>
canvas{border:2px solid #0f0;background:#1a2e1a;display:block;margin:20px auto}
body{background:#0a1a0a;text-align:center;color:#0f0;font-family:monospace}
button{background:#0a2a0a;color:#0f0;border:1px solid #0f0;padding:8px 16px;cursor:pointer;margin:5px}
</style></head><body><h1>Tower Defense</h1>
<button onclick="mode='tower'">Place Tower (50g)</button>
<button onclick="sendWave()">Send Wave</button>
<span id="info">Gold: 200 | Wave: 0 | Lives: 10</span>
<canvas id="td" width="500" height="400"></canvas><script>
const c=document.getElementById('td');const ctx=c.getContext('2d');
const path=[{x:0,y:200},{x:100,y:200},{x:100,y:100},{x:300,y:100},
{x:300,y:300},{x:450,y:300},{x:450,y:200},{x:500,y:200}];
let towers=[];let enemies=[];let projectiles=[];
let gold=200;let lives=10;let wave=0;let mode='';
c.addEventListener('click',e=>{if(mode==='tower'&&gold>=50){
let r=c.getBoundingClientRect();let x=e.clientX-r.left;let y=e.clientY-r.top;
towers.push({x:x,y:y,range:80,damage:1,cooldown:0});gold-=50;mode='';}});
function sendWave(){wave++;for(let i=0;i<wave*5;i++){
enemies.push({pathIdx:0,t:0,hp:3+wave,maxHp:3+wave,speed:1+wave*0.1,delay:i*20});}}
function lerp(a,b,t){return{x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t};}
function getEnemyPos(e){if(e.pathIdx>=path.length-1)return path[path.length-1];
return lerp(path[e.pathIdx],path[e.pathIdx+1],e.t);}
function update(){enemies.forEach(e=>{if(e.delay>0){e.delay--;return;}
e.t+=e.speed*0.01;if(e.t>=1){e.t=0;e.pathIdx++;
if(e.pathIdx>=path.length-1){e.hp=0;lives--;}}});
enemies=enemies.filter(e=>e.hp>0);
towers.forEach(tw=>{if(tw.cooldown>0){tw.cooldown--;return;}
for(let e of enemies){if(e.delay>0)continue;let pos=getEnemyPos(e);
let dx=tw.x-pos.x;let dy=tw.y-pos.y;if(Math.sqrt(dx*dx+dy*dy)<tw.range){
projectiles.push({x:tw.x,y:tw.y,tx:pos.x,ty:pos.y,target:e,dmg:tw.damage});
tw.cooldown=30;break;}}});
projectiles.forEach(p=>{p.x+=(p.tx-p.x)*0.2;p.y+=(p.ty-p.y)*0.2;
if(Math.abs(p.x-p.tx)<5&&Math.abs(p.y-p.ty)<5){p.target.hp-=p.dmg;
if(p.target.hp<=0)gold+=10;p.done=true;}});
projectiles=projectiles.filter(p=>!p.done);
document.getElementById('info').textContent=
'Gold: '+gold+' | Wave: '+wave+' | Lives: '+lives;}
function draw(){ctx.fillStyle='#1a2e1a';ctx.fillRect(0,0,500,400);
ctx.strokeStyle='#4a3';ctx.lineWidth=20;ctx.beginPath();ctx.moveTo(path[0].x,path[0].y);
path.forEach(p=>ctx.lineTo(p.x,p.y));ctx.stroke();
towers.forEach(tw=>{ctx.fillStyle='#00f';ctx.beginPath();ctx.arc(tw.x,tw.y,10,0,Math.PI*2);ctx.fill();
ctx.strokeStyle='rgba(0,0,255,0.2)';ctx.beginPath();ctx.arc(tw.x,tw.y,tw.range,0,Math.PI*2);ctx.stroke();});
enemies.forEach(e=>{if(e.delay>0)return;let pos=getEnemyPos(e);
ctx.fillStyle='#f00';ctx.beginPath();ctx.arc(pos.x,pos.y,8,0,Math.PI*2);ctx.fill();
ctx.fillStyle='#0f0';ctx.fillRect(pos.x-10,pos.y-15,20*(e.hp/e.maxHp),4);});
ctx.fillStyle='#ff0';projectiles.forEach(p=>{ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);ctx.fill();});}
function loop(){update();draw();requestAnimationFrame(loop);}loop();
</script></body></html>""",

    # ---- RACING GAME ----
    """<!DOCTYPE html><html><head><title>Racing</title><style>
canvas{border:2px solid #ff0;background:#333;display:block;margin:20px auto}
body{background:#1a1a00;text-align:center;color:#ff0;font-family:Arial}
</style></head><body><h1>Racing</h1><canvas id="race" width="300" height="500"></canvas>
<script>const c=document.getElementById('race');const ctx=c.getContext('2d');
let car={x:150,y:420,w:30,h:50};let obstacles=[];let score=0;let speed=4;let keys={};
document.addEventListener('keydown',e=>keys[e.key]=true);
document.addEventListener('keyup',e=>keys[e.key]=false);
function spawn(){let lane=Math.floor(Math.random()*3);
let x=50+lane*90;obstacles.push({x:x,y:-60,w:30,h:50,color:['#f00','#00f','#0f0'][lane]});}
function update(){score++;if(score%100===0)speed+=0.3;
if(keys['ArrowLeft'])car.x-=4;if(keys['ArrowRight'])car.x+=4;
car.x=Math.max(30,Math.min(240,car.x));
if(Math.random()<0.02)spawn();obstacles.forEach(o=>o.y+=speed);
obstacles=obstacles.filter(o=>o.y<560);
for(let o of obstacles){if(car.x<o.x+o.w&&car.x+car.w>o.x&&car.y<o.y+o.h&&car.y+car.h>o.y){
alert('Crash! Score: '+Math.floor(score/10));score=0;speed=4;obstacles=[];}}}
function draw(){ctx.fillStyle='#333';ctx.fillRect(0,0,300,500);
ctx.fillStyle='#2a2a2a';ctx.fillRect(30,0,240,500);
for(let i=0;i<10;i++){let y=((score*speed)%50+i*50)%500;
ctx.fillStyle='#ff0';ctx.fillRect(148,y,4,25);}
ctx.fillStyle='#fff';ctx.fillRect(car.x,car.y,car.w,car.h);
ctx.fillStyle='#0af';ctx.fillRect(car.x+5,car.y+5,20,15);
obstacles.forEach(o=>{ctx.fillStyle=o.color;ctx.fillRect(o.x,o.y,o.w,o.h);});
ctx.fillStyle='#ff0';ctx.font='16px Arial';ctx.fillText('Score: '+Math.floor(score/10),10,25);}
function loop(){update();draw();requestAnimationFrame(loop);}loop();
</script></body></html>""",

    # ---- WHACK-A-MOLE ----
    """<!DOCTYPE html><html><head><title>Whack-a-Mole</title><style>
body{background:#4a7c3f;text-align:center;color:#fff;font-family:Arial}
.grid{display:grid;grid-template-columns:repeat(3,100px);gap:15px;justify-content:center;margin:20px auto}
.hole{width:100px;height:100px;background:#3a2a0a;border:3px solid #2a1a00;border-radius:50%;
cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:40px;
transition:background 0.1s;user-select:none}
.hole.active{background:#8b6914;transform:scale(1.1)}
.hole:active{transform:scale(0.9)}
#info{font-size:22px;margin:10px;color:#ff0}
</style></head><body><h1>Whack-a-Mole!</h1><p id="info">Score: 0 | Time: 30</p>
<div class="grid" id="grid"></div><script>
const grid=document.getElementById('grid');let score=0;let timeLeft=30;let active=-1;
for(let i=0;i<9;i++){let hole=document.createElement('div');hole.className='hole';
hole.dataset.idx=i;hole.textContent='O';
hole.addEventListener('click',()=>{if(parseInt(hole.dataset.idx)===active){
score+=10;hole.classList.remove('active');hole.textContent='O';active=-1;}});
grid.appendChild(hole);}
const holes=document.querySelectorAll('.hole');
function showMole(){if(active>=0){holes[active].classList.remove('active');holes[active].textContent='O';}
active=Math.floor(Math.random()*9);holes[active].classList.add('active');holes[active].textContent='M';}
let moleInterval=setInterval(showMole,800);
let timer=setInterval(()=>{timeLeft--;
document.getElementById('info').textContent='Score: '+score+' | Time: '+timeLeft;
if(timeLeft<=0){clearInterval(moleInterval);clearInterval(timer);
alert('Time up! Final Score: '+score);score=0;timeLeft=30;
moleInterval=setInterval(showMole,800);timer=setInterval(arguments.callee,1000);}},1000);
showMole();</script></body></html>""",

    # ---- CANVAS DRAWING PARTICLES ----
    """<!DOCTYPE html><html><head><title>Particles</title><style>
canvas{border:none;background:#000;display:block;margin:0;cursor:crosshair}
body{margin:0;overflow:hidden;background:#000}
</style></head><body><canvas id="particles"></canvas><script>
const c=document.getElementById('particles');const ctx=c.getContext('2d');
c.width=window.innerWidth;c.height=window.innerHeight;
let particles=[];let mouseX=0;let mouseY=0;let hue=0;
c.addEventListener('mousemove',e=>{mouseX=e.clientX;mouseY=e.clientY;
for(let i=0;i<3;i++){particles.push({x:mouseX,y:mouseY,
vx:(Math.random()-0.5)*6,vy:(Math.random()-0.5)*6,
life:1,color:'hsl('+hue+',100%,50%)',size:Math.random()*4+2});}hue=(hue+2)%360;});
c.addEventListener('click',e=>{for(let i=0;i<50;i++){let angle=Math.random()*Math.PI*2;
let speed=Math.random()*8+2;particles.push({x:e.clientX,y:e.clientY,
vx:Math.cos(angle)*speed,vy:Math.sin(angle)*speed,
life:1,color:'hsl('+hue+',100%,'+Math.random()*50+50+'%)',size:Math.random()*6+2});}});
function update(){particles.forEach(p=>{p.x+=p.vx;p.y+=p.vy;p.vy+=0.05;
p.life-=0.015;p.size*=0.99;});particles=particles.filter(p=>p.life>0);}
function draw(){ctx.fillStyle='rgba(0,0,0,0.1)';ctx.fillRect(0,0,c.width,c.height);
particles.forEach(p=>{ctx.globalAlpha=p.life;ctx.fillStyle=p.color;
ctx.beginPath();ctx.arc(p.x,p.y,p.size,0,Math.PI*2);ctx.fill();});
ctx.globalAlpha=1;}
function loop(){update();draw();requestAnimationFrame(loop);}loop();
window.addEventListener('resize',()=>{c.width=window.innerWidth;c.height=window.innerHeight;});
</script></body></html>""",

    # ---- CLICK SPEED TEST ----
    """<!DOCTYPE html><html><head><title>Click Speed</title><style>
body{background:#1a1a2e;text-align:center;color:#fff;font-family:Arial;user-select:none}
#target{width:80px;height:80px;background:#e94560;border-radius:50%;position:absolute;
cursor:pointer;transition:none;display:flex;align-items:center;justify-content:center;
font-weight:bold;font-size:18px;color:#fff;box-shadow:0 0 20px #e94560}
#area{width:500px;height:400px;background:#16213e;margin:20px auto;position:relative;
border:2px solid #0f3460;border-radius:10px;overflow:hidden}
#stats{font-size:20px;margin:15px;color:#e94560}
</style></head><body><h1>Click Speed Test</h1><p id="stats">Click the targets! Score: 0</p>
<div id="area"><div id="target">GO</div></div><script>
const target=document.getElementById('target');const area=document.getElementById('area');
let score=0;let misses=0;let startTime=null;let clicks=0;
function moveTarget(){target.style.left=Math.random()*420+'px';
target.style.top=Math.random()*320+'px';
target.style.background='hsl('+Math.random()*360+',70%,50%)';}
moveTarget();target.addEventListener('click',e=>{e.stopPropagation();
if(!startTime)startTime=Date.now();score++;clicks++;moveTarget();updateStats();});
area.addEventListener('click',()=>{misses++;updateStats();});
function updateStats(){let elapsed=startTime?((Date.now()-startTime)/1000).toFixed(1):'0';
let cps=startTime?(clicks/((Date.now()-startTime)/1000)).toFixed(1):'0';
document.getElementById('stats').textContent=
'Score: '+score+' | Misses: '+misses+' | Time: '+elapsed+'s | CPS: '+cps;}
</script></body></html>""",
]


def augment_snippets(snippets, rng, factor=3):
    """Aumentar datos con variaciones de los snippets originales."""
    augmented = list(snippets)

    color_replacements = {
        '#0f0': '#2ecc71', '#f00': '#e74c3c', '#00f': '#3498db',
        '#ff0': '#f1c40f', '#0ff': '#1abc9c', '#f0f': '#9b59b6',
        '#fff': '#ecf0f1', '#333': '#2c3e50', '#111': '#1a1a2e',
    }
    alt_colors = {
        '#0f0': '#00ff88', '#f00': '#ff4444', '#00f': '#4488ff',
        '#ff0': '#ffaa00', '#0ff': '#00ffcc', '#f0f': '#ff44ff',
    }

    for snippet in snippets:
        # Variante con colores distintos
        v = snippet
        for old, new in color_replacements.items():
            if rng.random() > 0.5:
                v = v.replace(old, new)
        augmented.append(v)

        # Variante con otros colores
        v2 = snippet
        for old, new in alt_colors.items():
            if rng.random() > 0.5:
                v2 = v2.replace(old, new)
        augmented.append(v2)

        # Variante con dimensiones cambiadas
        v3 = snippet
        for dim in ['400', '500', '600']:
            if rng.random() > 0.5:
                new_dim = str(int(dim) + rng.choice([-100, -50, 50, 100]))
                v3 = v3.replace(f'width="{dim}"', f'width="{new_dim}"', 1)
        augmented.append(v3)

    rng.shuffle(augmented)
    return augmented


# =============================================================================
# Entrenamiento
# =============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("mztrain")

    TRAIN_MINUTES = 20
    CHECKPOINT_PATH = "zcodebert_trained.pt"
    SAVE_PATH = "zcodebert_htmlgames.pt"

    print("=" * 70)
    print("ZCodeBERT - Entrenamiento Videojuegos HTML5 (20 min)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # =========================================================================
    # 1. Cargar checkpoint existente
    # =========================================================================
    print(f"\n--- Cargando checkpoint: {CHECKPOINT_PATH} ---")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

    config_dict = checkpoint["config"]
    config = ZCodeBERTConfig.from_dict(config_dict)

    print(f"  Hidden size:     {config.hidden_size}")
    print(f"  Num layers:      {config.num_hidden_layers}")
    print(f"  Num heads:       {config.num_attention_heads}")
    print(f"  Rank:            {config.rank}")
    print(f"  Max seq len:     {config.max_position_embeddings}")

    model = ZCodeBERTForMLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Modelo cargado OK")

    stats = model.get_model_stats()
    print(f"  Parametros reales:       {stats['total_params']:,}")
    print(f"  Parametros equivalentes: {stats['equivalent_params']:,}")
    print(f"  Capas factorizadas:      {stats['factorized_layers']}")

    # =========================================================================
    # 2. Preparar dataset de juegos HTML5
    # =========================================================================
    print(f"\n--- Preparando dataset HTML5 Games ---")
    rng = random.Random(42)
    snippets = augment_snippets(HTML_GAME_SNIPPETS, rng, factor=3)
    print(f"  Snippets base:     {len(HTML_GAME_SNIPPETS)}")
    print(f"  Snippets aumentados: {len(snippets)}")

    tokenizer = CodeTokenizer(
        vocab_size=config.vocab_size,
        max_length=config.max_position_embeddings,
    )
    code_dataset = CodeDataset(
        snippets, tokenizer, max_length=config.max_position_embeddings,
    )
    mlm_dataset = MLMDataset(code_dataset, mask_prob=0.15, seed=42)

    n_train = int(0.85 * len(mlm_dataset))
    n_val = len(mlm_dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        mlm_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = 4 if device.type == "cpu" else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"  Train samples: {n_train}, Val samples: {n_val}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches por epoch: {len(train_loader)}")

    # =========================================================================
    # 3. Configurar optimizer y scheduler
    # =========================================================================
    lr = 2e-5 if device.type == "cuda" else 5e-5  # LR bajo para fine-tuning
    optimizer = ZCompressedAdam(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        compress_states=True,
    )

    rank_scheduler = ZRankScheduler(
        initial_rank=config.rank,
        max_rank=config.rank * 2,  # crecimiento conservador
        total_epochs=200,
        schedule=RankSchedule.EXPONENTIAL,
        growth_interval=50,
        growth_factor=1.5,
    )

    grad_compressor = ZGradientCompressor()

    # Warmup: incrementar LR gradualmente los primeros epochs
    warmup_epochs = 10
    base_lr = lr

    print(f"\n--- Configuracion entrenamiento ---")
    print(f"  Learning rate: {lr} (con warmup de {warmup_epochs} epochs)")
    print(f"  Duracion: {TRAIN_MINUTES} minutos")
    print(f"  Rango inicial: {config.rank}")
    print(f"  Rango maximo: {config.rank * 2}")
    print(f"  Schedule: EXPONENTIAL")

    # =========================================================================
    # 4. Training loop (20 minutos)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  ENTRENANDO... ({TRAIN_MINUTES} minutos)")
    print(f"{'='*70}\n")

    ZActivationCheckpoint.reset()
    best_val_loss = float("inf")
    start_time = time.time()
    max_seconds = TRAIN_MINUTES * 60
    epoch = 0
    total_batches = 0
    all_train_losses = []
    all_val_losses = []

    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            break

        epoch_start = time.time()

        # Crecimiento de rango
        new_rank = rank_scheduler.get_rank(epoch)
        if new_rank > rank_scheduler.current_rank:
            print(f"\n  >>> Crecimiento de rango: {rank_scheduler.current_rank} -> {new_rank}")
            model.grow_rank(new_rank)
            rank_scheduler.current_rank = new_rank
            optimizer = ZCompressedAdam(
                model.parameters(),
                lr=lr,
                weight_decay=0.01,
                compress_states=True,
            )

        # Ajustar LR con warmup
        if epoch < warmup_epochs:
            current_lr = base_lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

        # Train
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            # Check time limit
            if time.time() - start_time >= max_seconds:
                break

            masked_ids, attn_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            loss, logits = model(masked_ids, attention_mask=attn_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # clip mas agresivo
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            total_batches += 1

        if n_batches == 0:
            break

        train_loss = total_loss / n_batches
        all_train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                masked_ids, attn_mask, labels = [b.to(device) for b in batch]
                loss, _ = model(masked_ids, attention_mask=attn_mask, labels=labels)
                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / max(val_batches, 1)
        all_val_losses.append(val_loss)
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = max_seconds - elapsed

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = " * (best)"
            # Guardar mejor modelo
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "stats": model.get_model_stats(),
                "training_info": {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "rank": rank_scheduler.current_rank,
                    "total_batches": total_batches,
                    "elapsed_minutes": elapsed / 60,
                    "dataset": "html5_games",
                    "n_snippets": len(snippets),
                },
            }, SAVE_PATH)
        else:
            marker = ""

        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        rem_mins = int(remaining // 60)
        rem_secs = int(remaining % 60)

        print(
            f"  Epoch {epoch:3d} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"rank={rank_scheduler.current_rank} | "
            f"time={epoch_time:.1f}s | "
            f"[{mins:02d}:{secs:02d}/{TRAIN_MINUTES}:00 rem={rem_mins:02d}:{rem_secs:02d}]"
            f"{marker}"
        )

        epoch += 1

    total_time = time.time() - start_time

    # =========================================================================
    # 5. Resultados finales
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"  Tiempo total:        {total_time/60:.1f} minutos")
    print(f"  Epochs completados:  {epoch}")
    print(f"  Batches totales:     {total_batches}")
    print(f"  Mejor val_loss:      {best_val_loss:.4f}")
    print(f"  Rango final:         {rank_scheduler.current_rank}")

    if all_train_losses:
        print(f"  Train loss inicio:   {all_train_losses[0]:.4f}")
        print(f"  Train loss final:    {all_train_losses[-1]:.4f}")
        improvement = (1 - all_train_losses[-1] / all_train_losses[0]) * 100
        print(f"  Mejora:              {improvement:.1f}%")

    final_stats = model.get_model_stats()
    print(f"\n  Parametros finales:  {final_stats['total_params']:,}")
    print(f"  Rango final:         {final_stats['rank']}")

    opt_stats = optimizer.get_memory_stats()
    print(f"  Optimizer stats:     {opt_stats}")

    ckpt_stats = ZActivationCheckpoint.get_stats()
    print(f"  Activaciones comprimidas: {ckpt_stats['total_compressed']}")

    print(f"\n  Modelo guardado en: {SAVE_PATH}")
    print(f"{'='*70}")
    print(f"  Listo para generar videojuegos HTML5!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
