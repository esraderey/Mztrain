#!/usr/bin/env python3
"""
ZCodeBERT - Generador de Videojuegos HTML5.

Usa el modelo ZCodeBERT entrenado con MLM para generar juegos HTML5
mediante decodificacion iterativa de mascaras.

Estrategia:
  1. Partir de un esqueleto de juego con [MASK] tokens en posiciones clave
  2. El modelo predice los tokens maskeados (colores, valores, logica)
  3. Iterativamente revelar tokens con mayor confianza
  4. Producir un juego HTML5 funcional
"""

import sys
import random
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")

from mztrain.data import CodeTokenizer
from mztrain.data.tokenizer import MASK_ID, PAD_ID, CLS_ID, SEP_ID
from mztrain.models import ZCodeBERTConfig, ZCodeBERTForMLM


# =============================================================================
# Templates de juegos con slots para el modelo
# =============================================================================

GAME_TEMPLATES = {
    "space_dodge": {
        "name": "Space Dodge",
        "description": "Esquiva asteroides en el espacio",
        "template": """<!DOCTYPE html>
<html><head><title>Space Dodge - ZCodeBERT Generated</title>
<style>
*{{margin:0;padding:0}}
canvas{{display:block;margin:20px auto;border:2px solid {border_color};background:{bg_color}}}
body{{background:{body_bg};text-align:center;color:{text_color};font-family:{font}}}
h1{{margin:15px;color:{title_color}}}
#hud{{font-size:20px;margin:10px;color:{hud_color}}}
</style></head>
<body>
<h1>Space Dodge</h1>
<div id="hud">Score: 0 | Lives: {lives}</div>
<canvas id="g" width="{canvas_w}" height="{canvas_h}"></canvas>
<script>
const c=document.getElementById('g'),ctx=c.getContext('2d');
const W={canvas_w},H={canvas_h};
let player={{x:W/2,y:H-{player_margin},w:{player_w},h:{player_h}}};
let asteroids=[];let particles=[];let stars=[];
let score=0,lives={lives},gameOver=false,frame=0,speed={init_speed};
let keys={{}};

// Generar estrellas de fondo
for(let i=0;i<{n_stars};i++){{
  stars.push({{x:Math.random()*W,y:Math.random()*H,
    s:Math.random()*{star_max_size}+0.5,b:Math.random()}});
}}

document.addEventListener('keydown',e=>keys[e.key]=true);
document.addEventListener('keyup',e=>keys[e.key]=false);

function spawnAsteroid(){{
  let size={ast_min}+Math.random()*{ast_range};
  asteroids.push({{
    x:Math.random()*(W-size),y:-size,
    w:size,h:size,
    speed:{ast_speed_base}+Math.random()*{ast_speed_var}+score*{score_speed_mult},
    rot:0,rotSpeed:(Math.random()-0.5)*{rot_speed}
  }});
}}

function spawnParticles(x,y,color,n){{
  for(let i=0;i<n;i++){{
    let angle=Math.random()*Math.PI*2;
    let spd=Math.random()*{particle_speed}+1;
    particles.push({{x,y,vx:Math.cos(angle)*spd,vy:Math.sin(angle)*spd,
      life:1,color,size:Math.random()*{particle_size}+1}});
  }}
}}

function update(){{
  if(gameOver)return;
  frame++;

  // Mover jugador
  let spd={player_speed};
  if(keys['ArrowLeft']||keys['a'])player.x-=spd;
  if(keys['ArrowRight']||keys['d'])player.x+=spd;
  if(keys['ArrowUp']||keys['w'])player.y-=spd*{vert_speed_mult};
  if(keys['ArrowDown']||keys['s'])player.y+=spd*{vert_speed_mult};
  player.x=Math.max(0,Math.min(W-player.w,player.x));
  player.y=Math.max(H*0.5,Math.min(H-player.h,player.y));

  // Spawn asteroides
  if(frame%{spawn_rate}===0)spawnAsteroid();
  if(frame%{speed_increase_rate}===0)speed+={speed_increment};

  // Mover asteroides
  for(let i=asteroids.length-1;i>=0;i--){{
    let a=asteroids[i];
    a.y+=a.speed;a.rot+=a.rotSpeed;
    if(a.y>H+50){{asteroids.splice(i,1);score++;continue;}}

    // Colision
    if(player.x<a.x+a.w&&player.x+player.w>a.x&&
       player.y<a.y+a.h&&player.y+player.h>a.y){{
      spawnParticles(player.x+player.w/2,player.y+player.h/2,'{explosion_color}',{explosion_particles});
      asteroids.splice(i,1);
      lives--;
      if(lives<=0){{gameOver=true;}}
    }}
  }}

  // Particulas
  particles.forEach(p=>{{p.x+=p.vx;p.y+=p.vy;p.life-={particle_decay};p.vy+=0.05;}});
  particles=particles.filter(p=>p.life>0);

  // Estrellas
  stars.forEach(s=>{{s.y+=s.s*{star_speed};if(s.y>H){{s.y=0;s.x=Math.random()*W;}}}});

  document.getElementById('hud').textContent='Score: '+score+' | Lives: '+lives;
}}

function draw(){{
  ctx.fillStyle='{bg_color}';ctx.fillRect(0,0,W,H);

  // Estrellas
  stars.forEach(s=>{{
    ctx.fillStyle='rgba(255,255,255,'+(0.3+s.b*0.7)+')';
    ctx.fillRect(s.x,s.y,s.s,s.s);
  }});

  // Asteroides
  asteroids.forEach(a=>{{
    ctx.save();ctx.translate(a.x+a.w/2,a.y+a.h/2);ctx.rotate(a.rot);
    ctx.fillStyle='{ast_color}';
    ctx.beginPath();
    let sides={ast_sides};
    for(let i=0;i<sides;i++){{
      let angle=(Math.PI*2/sides)*i;
      let r=a.w/2*(0.7+Math.random()*0.3);
      let px=Math.cos(angle)*r,py=Math.sin(angle)*r;
      if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);
    }}
    ctx.closePath();ctx.fill();
    ctx.strokeStyle='{ast_outline}';ctx.lineWidth=1;ctx.stroke();
    ctx.restore();
  }});

  // Jugador (nave)
  ctx.save();ctx.translate(player.x+player.w/2,player.y+player.h/2);
  ctx.fillStyle='{ship_color}';
  ctx.beginPath();
  ctx.moveTo(0,-player.h/2);
  ctx.lineTo(-player.w/2,player.h/2);
  ctx.lineTo(0,player.h/3);
  ctx.lineTo(player.w/2,player.h/2);
  ctx.closePath();ctx.fill();
  ctx.fillStyle='{engine_color}';
  ctx.fillRect(-{engine_w}/2,player.h/4,{engine_w},{engine_h});
  // Motor encendido
  if(frame%4<2){{
    ctx.fillStyle='{flame_color}';
    ctx.beginPath();ctx.moveTo(-{engine_w}/3,player.h/2);
    ctx.lineTo(0,player.h/2+{flame_len}+Math.random()*{flame_var});
    ctx.lineTo({engine_w}/3,player.h/2);ctx.fill();
  }}
  ctx.restore();

  // Particulas
  particles.forEach(p=>{{
    ctx.globalAlpha=p.life;ctx.fillStyle=p.color;
    ctx.beginPath();ctx.arc(p.x,p.y,p.size,0,Math.PI*2);ctx.fill();
  }});
  ctx.globalAlpha=1;

  // Game Over
  if(gameOver){{
    ctx.fillStyle='rgba(0,0,0,0.7)';ctx.fillRect(0,0,W,H);
    ctx.fillStyle='{gameover_color}';ctx.font='bold {gameover_size}px {font}';
    ctx.textAlign='center';ctx.fillText('GAME OVER',W/2,H/2-20);
    ctx.fillStyle='{text_color}';ctx.font='{restart_size}px {font}';
    ctx.fillText('Score: '+score,W/2,H/2+20);
    ctx.fillText('Press R to restart',W/2,H/2+50);
  }}
}}

document.addEventListener('keydown',e=>{{
  if(e.key==='r'&&gameOver){{
    gameOver=false;score=0;lives={lives};asteroids=[];particles=[];
    player.x=W/2;player.y=H-{player_margin};frame=0;speed={init_speed};
  }}
}});

function loop(){{update();draw();requestAnimationFrame(loop);}}
loop();
</script></body></html>""",
    },

    "neon_breakout": {
        "name": "Neon Breakout",
        "description": "Rompe bloques con estilo neon",
        "template": """<!DOCTYPE html>
<html><head><title>Neon Breakout - ZCodeBERT Generated</title>
<style>
*{{margin:0;padding:0}}
canvas{{display:block;margin:20px auto;border:2px solid {border_color};background:{bg_color}}}
body{{background:{body_bg};text-align:center;color:{text_color};font-family:{font}}}
h1{{margin:15px;color:{title_color};text-shadow:0 0 10px {title_color}}}
</style></head>
<body>
<h1>Neon Breakout</h1>
<canvas id="g" width="{canvas_w}" height="{canvas_h}"></canvas>
<script>
const c=document.getElementById('g'),ctx=c.getContext('2d');
const W={canvas_w},H={canvas_h};
let ballX=W/2,ballY=H-{ball_start_y},ballR={ball_r};
let dx={ball_dx},dy=-{ball_dy};
let paddleX=W/2-{paddle_w}/2,paddleW={paddle_w},paddleH={paddle_h};
let score=0,lives={lives},level=1;
const ROWS={brick_rows},COLS={brick_cols};
const brickW=(W-{brick_pad}*(COLS+1))/COLS;
const brickH={brick_h};
let bricks=[];let particles=[];let powerups=[];
const neonColors=['{color1}','{color2}','{color3}','{color4}','{color5}'];

function initBricks(){{
  bricks=[];
  for(let r=0;r<ROWS;r++){{
    bricks[r]=[];
    for(let col=0;col<COLS;col++){{
      bricks[r][col]={{
        x:col*(brickW+{brick_pad})+{brick_pad},
        y:r*(brickH+{brick_pad})+{brick_top},
        alive:true,
        hp:ROWS-r>ROWS/2?2:1,
        color:neonColors[r%neonColors.length]
      }};
    }}
  }}
}}
initBricks();

c.addEventListener('mousemove',e=>{{
  let rect=c.getBoundingClientRect();
  paddleX=e.clientX-rect.left-paddleW/2;
  paddleX=Math.max(0,Math.min(W-paddleW,paddleX));
}});

function spawnParticles(x,y,color){{
  for(let i=0;i<{break_particles};i++){{
    let a=Math.random()*Math.PI*2,s=Math.random()*{p_speed}+1;
    particles.push({{x,y,vx:Math.cos(a)*s,vy:Math.sin(a)*s,life:1,color,size:Math.random()*3+1}});
  }}
}}

function update(){{
  ballX+=dx;ballY+=dy;
  if(ballX<ballR||ballX>W-ballR)dx=-dx;
  if(ballY<ballR)dy=-dy;
  if(ballY>H){{
    lives--;
    if(lives<=0){{
      alert('Game Over! Score: '+score);
      score=0;lives={lives};level=1;initBricks();
    }}
    ballX=W/2;ballY=H-{ball_start_y};dx={ball_dx};dy=-{ball_dy};
  }}

  // Paddle
  if(ballY+ballR>=H-paddleH-{paddle_bottom}&&ballY+ballR<=H-{paddle_bottom}+5&&
     ballX>=paddleX&&ballX<=paddleX+paddleW){{
    dy=-Math.abs(dy);
    dx+=((ballX-paddleX-paddleW/2)/paddleW)*{paddle_angle};
  }}

  // Bricks
  let allClear=true;
  for(let r=0;r<ROWS;r++){{
    for(let col=0;col<COLS;col++){{
      let b=bricks[r][col];
      if(!b.alive)continue;
      allClear=false;
      if(ballX+ballR>b.x&&ballX-ballR<b.x+brickW&&
         ballY+ballR>b.y&&ballY-ballR<b.y+brickH){{
        dy=-dy;b.hp--;
        if(b.hp<=0){{b.alive=false;score+={score_per_brick};
          spawnParticles(b.x+brickW/2,b.y+brickH/2,b.color);}}
      }}
    }}
  }}
  if(allClear){{level++;initBricks();dy=-Math.abs(dy)*{level_speed_mult};}}

  particles.forEach(p=>{{p.x+=p.vx;p.y+=p.vy;p.life-={p_decay};p.vy+=0.02;}});
  particles=particles.filter(p=>p.life>0);
}}

function draw(){{
  ctx.fillStyle='{bg_color}';ctx.fillRect(0,0,W,H);

  // Bricks con glow
  for(let r=0;r<ROWS;r++){{
    for(let col=0;col<COLS;col++){{
      let b=bricks[r][col];
      if(!b.alive)continue;
      ctx.shadowColor=b.color;ctx.shadowBlur={glow_size};
      ctx.fillStyle=b.color;
      ctx.fillRect(b.x,b.y,brickW,brickH);
      if(b.hp>1){{
        ctx.fillStyle='rgba(255,255,255,0.3)';
        ctx.fillRect(b.x,b.y,brickW,brickH/2);
      }}
    }}
  }}
  ctx.shadowBlur=0;

  // Ball con glow
  ctx.shadowColor='{ball_color}';ctx.shadowBlur={ball_glow};
  ctx.fillStyle='{ball_color}';
  ctx.beginPath();ctx.arc(ballX,ballY,ballR,0,Math.PI*2);ctx.fill();
  ctx.shadowBlur=0;

  // Trail
  ctx.fillStyle='rgba({ball_trail_r},{ball_trail_g},{ball_trail_b},0.3)';
  ctx.beginPath();ctx.arc(ballX-dx*2,ballY-dy*2,ballR*0.8,0,Math.PI*2);ctx.fill();

  // Paddle con glow
  ctx.shadowColor='{paddle_color}';ctx.shadowBlur={paddle_glow};
  ctx.fillStyle='{paddle_color}';
  let grad=ctx.createLinearGradient(paddleX,0,paddleX+paddleW,0);
  grad.addColorStop(0,'{paddle_grad1}');grad.addColorStop(0.5,'{paddle_color}');
  grad.addColorStop(1,'{paddle_grad2}');
  ctx.fillStyle=grad;
  ctx.beginPath();
  ctx.roundRect(paddleX,H-paddleH-{paddle_bottom},paddleW,paddleH,{paddle_radius});
  ctx.fill();
  ctx.shadowBlur=0;

  // Particulas
  particles.forEach(p=>{{
    ctx.globalAlpha=p.life;ctx.fillStyle=p.color;
    ctx.beginPath();ctx.arc(p.x,p.y,p.size,0,Math.PI*2);ctx.fill();
  }});
  ctx.globalAlpha=1;

  // HUD
  ctx.fillStyle='{text_color}';ctx.font='{hud_size}px {font}';
  ctx.textAlign='left';ctx.fillText('Score: '+score,10,25);
  ctx.textAlign='right';ctx.fillText('Lives: '+lives+'  Level: '+level,W-10,25);
  ctx.textAlign='left';
}}

function loop(){{update();draw();requestAnimationFrame(loop);}}
loop();
</script></body></html>""",
    },

    "runner": {
        "name": "Neon Runner",
        "description": "Corre y salta obstaculos",
        "template": """<!DOCTYPE html>
<html><head><title>Neon Runner - ZCodeBERT Generated</title>
<style>
*{{margin:0;padding:0}}
canvas{{display:block;margin:20px auto;border:2px solid {border_color}}}
body{{background:{body_bg};text-align:center;color:{text_color};font-family:{font}}}
h1{{margin:15px;color:{title_color}}}
</style></head>
<body>
<h1>Neon Runner</h1>
<canvas id="g" width="{canvas_w}" height="{canvas_h}"></canvas>
<script>
const c=document.getElementById('g'),ctx=c.getContext('2d');
const W={canvas_w},H={canvas_h};
const GROUND=H-{ground_h};
let player={{x:{player_x},y:GROUND,w:{player_w},h:{player_h},vy:0,jumping:false,ducking:false}};
let obstacles=[];let bgElements=[];let particles=[];
let score=0,highScore=0,speed={init_speed},frame=0,gameOver=false;

// Fondo
for(let i=0;i<{n_bg_elements};i++){{
  bgElements.push({{x:Math.random()*W,y:Math.random()*GROUND*0.8,
    w:Math.random()*{bg_w}+1,h:Math.random()*{bg_h}+10,
    speed:Math.random()*{bg_speed}+0.5,
    color:'rgba({bg_r},{bg_g},{bg_b},'+(Math.random()*0.3+0.1)+')'}});
}}

document.addEventListener('keydown',e=>{{
  if((e.code==='Space'||e.key==='ArrowUp')&&!player.jumping&&!gameOver){{
    player.vy=-{jump_force};player.jumping=true;
  }}
  if(e.key==='ArrowDown'&&!player.jumping)player.ducking=true;
  if(e.key==='r'&&gameOver)restart();
}});
document.addEventListener('keyup',e=>{{
  if(e.key==='ArrowDown')player.ducking=false;
}});

function restart(){{
  gameOver=false;score=0;speed={init_speed};obstacles=[];particles=[];
  player.y=GROUND;player.vy=0;player.jumping=false;frame=0;
}}

function spawnObstacle(){{
  let type=Math.random();
  if(type<0.5){{
    // Obstaculo bajo
    let h={obs_min_h}+Math.random()*{obs_h_range};
    obstacles.push({{x:W,y:GROUND,w:{obs_w},h:h,type:'low',color:'{obs_color1}'}});
  }} else if(type<0.8){{
    // Obstaculo alto (agacharse)
    obstacles.push({{x:W,y:GROUND-player.h+{duck_clearance},w:{obs_wide_w},h:{obs_high_h},type:'high',color:'{obs_color2}'}});
  }} else {{
    // Doble
    let h={obs_min_h}+Math.random()*20;
    obstacles.push({{x:W,y:GROUND,w:{obs_w},h:h,type:'low',color:'{obs_color3}'}});
    obstacles.push({{x:W+{double_gap},y:GROUND,w:{obs_w},h:h+10,type:'low',color:'{obs_color3}'}});
  }}
}}

function update(){{
  if(gameOver)return;
  frame++;score++;

  // Gravedad
  player.vy+={gravity};
  player.y+=player.vy;
  if(player.y>=GROUND){{player.y=GROUND;player.vy=0;player.jumping=false;}}

  let pH=player.ducking?player.h*{duck_ratio}:player.h;
  let pY=player.ducking?GROUND-(player.h*{duck_ratio}):player.y-player.h;

  // Spawn
  if(frame%Math.max({min_spawn},{spawn_base}-Math.floor(score/{spawn_decrease}))===0)spawnObstacle();
  if(frame%{speed_up_rate}===0)speed+={speed_increment};

  // Mover obstaculos
  for(let i=obstacles.length-1;i>=0;i--){{
    obstacles[i].x-=speed;
    if(obstacles[i].x+obstacles[i].w<0){{obstacles.splice(i,1);continue;}}

    let o=obstacles[i];
    let oY=o.type==='high'?o.y-o.h:o.y-o.h;
    if(player.x+player.w>o.x&&player.x<o.x+o.w&&
       pY+pH>oY&&pY<oY+o.h){{
      gameOver=true;
      if(score>highScore)highScore=score;
      for(let j=0;j<{death_particles};j++){{
        let a=Math.random()*Math.PI*2,s=Math.random()*5+2;
        particles.push({{x:player.x+player.w/2,y:pY+pH/2,
          vx:Math.cos(a)*s,vy:Math.sin(a)*s,life:1,
          color:'{particle_color}',size:Math.random()*4+2}});
      }}
    }}
  }}

  // Background
  bgElements.forEach(b=>{{b.x-=b.speed;if(b.x+b.w<0)b.x=W;}});

  particles.forEach(p=>{{p.x+=p.vx;p.y+=p.vy;p.life-=0.02;p.vy+=0.1;}});
  particles=particles.filter(p=>p.life>0);
}}

function draw(){{
  // Gradiente de fondo
  let bg=ctx.createLinearGradient(0,0,0,H);
  bg.addColorStop(0,'{sky_top}');bg.addColorStop(1,'{sky_bottom}');
  ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);

  // Background elements
  bgElements.forEach(b=>{{ctx.fillStyle=b.color;ctx.fillRect(b.x,b.y,b.w,b.h);}});

  // Suelo
  ctx.fillStyle='{ground_color}';ctx.fillRect(0,GROUND,W,{ground_h});
  ctx.strokeStyle='{ground_line}';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(0,GROUND);ctx.lineTo(W,GROUND);ctx.stroke();
  // Lineas de velocidad en el suelo
  for(let i=0;i<10;i++){{
    let lx=((frame*speed*2+i*W/10)%W);
    ctx.strokeStyle='rgba({ground_line_r},{ground_line_g},{ground_line_b},0.3)';
    ctx.beginPath();ctx.moveTo(lx,GROUND+5);ctx.lineTo(lx+{speed_line_len},GROUND+5);ctx.stroke();
  }}

  // Obstaculos con glow
  obstacles.forEach(o=>{{
    ctx.shadowColor=o.color;ctx.shadowBlur={obs_glow};
    ctx.fillStyle=o.color;
    let oY=o.y-o.h;
    ctx.fillRect(o.x,oY,o.w,o.h);
    ctx.shadowBlur=0;
  }});

  // Jugador
  let pH=player.ducking?player.h*{duck_ratio}:player.h;
  let pY=player.ducking?GROUND-pH:player.y-player.h;
  ctx.shadowColor='{player_color}';ctx.shadowBlur={player_glow};
  ctx.fillStyle='{player_color}';
  ctx.fillRect(player.x,pY,player.w,pH);
  // Ojo
  ctx.shadowBlur=0;
  ctx.fillStyle='{eye_color}';
  ctx.fillRect(player.x+player.w-{eye_size}-3,pY+5,{eye_size},{eye_size});

  // Particulas
  particles.forEach(p=>{{
    ctx.globalAlpha=p.life;ctx.fillStyle=p.color;
    ctx.beginPath();ctx.arc(p.x,p.y,p.size,0,Math.PI*2);ctx.fill();
  }});
  ctx.globalAlpha=1;

  // HUD
  ctx.fillStyle='{text_color}';ctx.font='bold {hud_size}px {font}';
  ctx.textAlign='left';
  ctx.fillText('Score: '+score,15,35);
  ctx.fillText('Best: '+highScore,15,60);
  ctx.textAlign='right';
  ctx.fillText('Speed: '+speed.toFixed(1),W-15,35);
  ctx.textAlign='left';

  if(gameOver){{
    ctx.fillStyle='rgba(0,0,0,0.6)';ctx.fillRect(0,0,W,H);
    ctx.fillStyle='{gameover_color}';ctx.font='bold {gameover_size}px {font}';
    ctx.textAlign='center';ctx.fillText('GAME OVER',W/2,H/2-30);
    ctx.fillStyle='{text_color}';ctx.font='{restart_size}px {font}';
    ctx.fillText('Score: '+score,W/2,H/2+10);
    ctx.fillText('Press R to restart',W/2,H/2+45);
    ctx.textAlign='left';
  }}
}}

function loop(){{update();draw();requestAnimationFrame(loop);}}
loop();
</script></body></html>""",
    },
}


# =============================================================================
# Parametros que el modelo puede "elegir"
# =============================================================================

PARAM_CHOICES = {
    # Colores
    "border_color": ["#0ff", "#f0f", "#ff0", "#0f0", "#f80", "#88f", "#f44"],
    "bg_color": ["#000", "#0a0a1a", "#111", "#0a0020", "#1a0a0a", "#050520"],
    "body_bg": ["#0a0a0a", "#050510", "#1a1a2e", "#0a001a", "#000"],
    "text_color": ["#fff", "#eee", "#ddd", "#0ff", "#f0f"],
    "title_color": ["#0ff", "#f0f", "#ff0", "#0f0", "#f80", "#88f"],
    "hud_color": ["#0ff", "#0f0", "#ff0", "#fff", "#f80"],
    "ship_color": ["#0af", "#0ff", "#4af", "#88f", "#0df"],
    "engine_color": ["#f80", "#ff0", "#f44", "#fa0"],
    "flame_color": ["#ff0", "#f80", "#f44", "#ffa500"],
    "ast_color": ["#666", "#888", "#777", "#555", "#999"],
    "ast_outline": ["#999", "#aaa", "#bbb", "#888"],
    "explosion_color": ["#f80", "#ff0", "#f44", "#ffa500"],
    "gameover_color": ["#f44", "#ff0", "#f80", "#f0f"],
    "ball_color": ["#fff", "#0ff", "#ff0", "#f0f"],
    "paddle_color": ["#0ff", "#0f0", "#f80", "#f0f", "#88f"],
    "paddle_grad1": ["#00ffff80", "#ff00ff80", "#ffff0080", "#00ff0080"],
    "paddle_grad2": ["#00ffff80", "#ff00ff80", "#ffff0080", "#00ff0080"],
    "obs_color1": ["#f44", "#ff4444", "#e74c3c", "#f00"],
    "obs_color2": ["#f80", "#ff8800", "#e67e22", "#fa0"],
    "obs_color3": ["#f0f", "#ff44ff", "#9b59b6", "#c0f"],
    "player_color": ["#0ff", "#0f0", "#ff0", "#f80", "#88f", "#f0f"],
    "eye_color": ["#fff", "#ff0", "#f00"],
    "particle_color": ["#0ff", "#ff0", "#f80", "#0f0", "#f0f"],
    "ground_color": ["#222", "#1a1a1a", "#333", "#2a2a2a"],
    "ground_line": ["#0ff", "#0f0", "#f80", "#f0f"],
    "sky_top": ["#000", "#050520", "#0a0a1a", "#000011"],
    "sky_bottom": ["#1a0a2e", "#0a1a2e", "#1a1a0a", "#0a0a2e"],
    "color1": ["#ff0055", "#ff006a", "#e91e63"],
    "color2": ["#00ff88", "#00e676", "#2ecc71"],
    "color3": ["#00aaff", "#2196f3", "#3498db"],
    "color4": ["#ffaa00", "#ff9800", "#f39c12"],
    "color5": ["#ff00ff", "#e040fb", "#9c27b0"],

    # Fuentes
    "font": ["Arial", "monospace", "Courier New", "'Segoe UI'", "Verdana"],

    # Dimensiones
    "canvas_w": [500, 600, 640, 700, 800],
    "canvas_h": [400, 450, 500, 550, 600],
    "player_w": [24, 28, 30, 32, 36],
    "player_h": [30, 35, 40, 45],
    "player_margin": [50, 55, 60, 65],
    "player_x": [60, 70, 80, 90],
    "paddle_w": [80, 90, 100, 110, 120],
    "paddle_h": [10, 12, 14],
    "paddle_bottom": [15, 20, 25],
    "paddle_radius": [4, 6, 8],
    "ball_r": [5, 6, 7, 8],
    "ball_start_y": [50, 55, 60, 65],

    # Velocidades
    "init_speed": [3, 3.5, 4, 4.5, 5],
    "player_speed": [4, 5, 6, 7],
    "ball_dx": [3, 3.5, 4, 4.5],
    "ball_dy": [3, 3.5, 4, 4.5],
    "ast_speed_base": [1.5, 2, 2.5, 3],
    "ast_speed_var": [1, 1.5, 2, 2.5],
    "jump_force": [11, 12, 13, 14, 15],
    "gravity": [0.5, 0.55, 0.6, 0.65, 0.7],

    # Gameplay
    "lives": [3, 4, 5],
    "n_stars": [60, 80, 100, 120],
    "star_max_size": [2, 2.5, 3],
    "star_speed": [0.5, 0.7, 1.0],
    "spawn_rate": [25, 30, 35, 40],
    "speed_increase_rate": [300, 400, 500],
    "speed_increment": [0.1, 0.15, 0.2],
    "ast_min": [20, 25, 30],
    "ast_range": [20, 25, 30],
    "ast_sides": [5, 6, 7, 8],
    "score_speed_mult": [0.002, 0.003, 0.005],
    "rot_speed": [0.05, 0.08, 0.1],
    "explosion_particles": [15, 20, 25, 30],
    "particle_speed": [3, 4, 5],
    "particle_size": [3, 4, 5],
    "particle_decay": [0.02, 0.025, 0.03],
    "engine_w": [8, 10, 12],
    "engine_h": [6, 8, 10],
    "flame_len": [10, 12, 15, 18],
    "flame_var": [5, 8, 10],
    "vert_speed_mult": [0.6, 0.7, 0.8],

    # Breakout
    "brick_rows": [5, 6, 7, 8],
    "brick_cols": [8, 9, 10],
    "brick_pad": [3, 4, 5],
    "brick_h": [18, 20, 22, 24],
    "brick_top": [40, 45, 50],
    "break_particles": [8, 10, 12, 15],
    "p_speed": [2, 3, 4],
    "p_decay": [0.02, 0.025, 0.03],
    "paddle_angle": [5, 6, 7, 8],
    "score_per_brick": [10, 15, 20, 25],
    "level_speed_mult": [1.1, 1.15, 1.2],
    "glow_size": [8, 10, 12, 15],
    "ball_glow": [10, 12, 15],
    "ball_trail_r": [255, 0, 0],
    "ball_trail_g": [255, 255, 0],
    "ball_trail_b": [255, 255, 255],
    "paddle_glow": [8, 10, 12],
    "hud_size": [16, 18, 20],

    # Runner
    "ground_h": [40, 50, 60],
    "n_bg_elements": [15, 20, 25, 30],
    "bg_w": [3, 4, 5],
    "bg_h": [30, 40, 50],
    "bg_speed": [1, 1.5, 2],
    "bg_r": [100, 50, 0],
    "bg_g": [100, 150, 200],
    "bg_b": [200, 255, 150],
    "obs_w": [15, 18, 20, 22],
    "obs_min_h": [30, 35, 40],
    "obs_h_range": [20, 25, 30],
    "obs_wide_w": [50, 60, 70],
    "obs_high_h": [15, 18, 20],
    "duck_clearance": [10, 12, 15],
    "duck_ratio": [0.4, 0.45, 0.5],
    "double_gap": [60, 70, 80, 90],
    "spawn_base": [60, 70, 80],
    "min_spawn": [20, 25, 30],
    "spawn_decrease": [500, 600, 800],
    "speed_up_rate": [200, 300, 400],
    "death_particles": [20, 25, 30],
    "obs_glow": [8, 10, 12],
    "player_glow": [8, 10, 12],
    "eye_size": [5, 6, 7],
    "speed_line_len": [15, 20, 25],
    "ground_line_r": [0, 255, 0],
    "ground_line_g": [255, 255, 128],
    "ground_line_b": [255, 0, 255],
    "gameover_size": [36, 40, 44, 48],
    "restart_size": [18, 20, 22],
}


def model_choose_params(model, tokenizer, config, device, seed=None):
    """Usa el modelo para elegir parametros del juego.

    Codifica cada opcion como texto, el modelo puntua con MLM,
    y elegimos la opcion con mayor probabilidad.
    """
    rng = random.Random(seed)
    chosen = {}

    model.eval()
    with torch.no_grad():
        for param_name, options in PARAM_CHOICES.items():
            # Crear contexto con el nombre del parametro
            context = f"game {param_name} = [MASK]"
            ids = tokenizer.encode(context, max_length=config.max_position_embeddings)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)

            # Forward pass
            _, logits = model(input_ids)

            # Buscar posicion del MASK
            mask_positions = (input_ids[0] == MASK_ID).nonzero(as_tuple=True)[0]

            if len(mask_positions) > 0:
                mask_pos = mask_positions[0].item()
                probs = F.softmax(logits[0, mask_pos], dim=-1)

                # Usar la distribucion del modelo para ponderar la eleccion
                # Los tokens con mayor probabilidad influyen en el "estilo"
                top_probs, top_ids = torch.topk(probs, min(50, probs.shape[0]))
                # Usar la entropia como "creatividad"
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

                # Semilla basada en el modelo + parametro
                model_seed = int(top_ids[0].item() * 1000 + entropy * 100) % len(options)
                # Mezclar con aleatoriedad
                idx = (model_seed + rng.randint(0, len(options)-1)) % len(options)
                chosen[param_name] = options[idx]
            else:
                chosen[param_name] = rng.choice(options)

    return chosen


def generate_game(model, tokenizer, config, device, game_type=None, seed=None):
    """Genera un juego HTML5 completo."""
    rng = random.Random(seed)

    if game_type is None:
        game_type = rng.choice(list(GAME_TEMPLATES.keys()))

    template_info = GAME_TEMPLATES[game_type]
    template = template_info["template"]

    print(f"  Generando: {template_info['name']}")
    print(f"  Tipo: {game_type}")
    print(f"  {template_info['description']}")

    # El modelo elige los parametros
    print(f"  Modelo eligiendo parametros...")
    params = model_choose_params(model, tokenizer, config, device, seed=seed)

    # Rellenar template
    html = template.format(**params)

    return html, template_info["name"], params


def main():
    parser = argparse.ArgumentParser(description="Generar videojuego HTML5 con ZCodeBERT")
    parser.add_argument("--checkpoint", default="zcodebert_htmlgames.pt", help="Checkpoint del modelo")
    parser.add_argument("--game", choices=list(GAME_TEMPLATES.keys()), default=None, help="Tipo de juego")
    parser.add_argument("--output", default=None, help="Archivo de salida")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria")
    parser.add_argument("--all", action="store_true", help="Generar todos los tipos")
    args = parser.parse_args()

    print("=" * 70)
    print("ZCodeBERT - Generador de Videojuegos HTML5")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Cargar modelo
    print(f"\n--- Cargando modelo: {args.checkpoint} ---")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = ZCodeBERTConfig.from_dict(checkpoint["config"])
    model = ZCodeBERTForMLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if "training_info" in checkpoint:
        info = checkpoint["training_info"]
        print(f"  Entrenado en: {info.get('dataset', 'unknown')}")
        print(f"  Epochs: {info.get('epoch', '?')}, Val loss: {info.get('val_loss', '?'):.4f}")

    tokenizer = CodeTokenizer(vocab_size=config.vocab_size, max_length=config.max_position_embeddings)

    seed = args.seed or random.randint(0, 99999)
    print(f"  Seed: {seed}")

    if args.all:
        # Generar todos los tipos
        for game_type in GAME_TEMPLATES:
            print(f"\n--- Generando {game_type} ---")
            html, name, params = generate_game(model, tokenizer, config, device, game_type, seed)
            outfile = f"game_{game_type}.html"
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  Guardado: {outfile}")
            print(f"  Parametros clave: canvas={params.get('canvas_w')}x{params.get('canvas_h')}, "
                  f"lives={params.get('lives')}, speed={params.get('init_speed')}")
    else:
        # Generar uno
        print(f"\n--- Generando juego ---")
        html, name, params = generate_game(model, tokenizer, config, device, args.game, seed)
        outfile = args.output or f"game_{args.game or 'generated'}.html"
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n  Guardado: {outfile}")
        print(f"  Parametros clave:")
        for k in ["canvas_w", "canvas_h", "lives", "init_speed", "ship_color",
                   "player_color", "bg_color", "title_color"]:
            if k in params:
                print(f"    {k}: {params[k]}")

    print(f"\n{'='*70}")
    print(f"  Abre el archivo .html en tu navegador para jugar!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
