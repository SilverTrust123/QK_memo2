#include <SPI.h>
  #include <Wire.h>
  #include <Adafruit_GFX.h>
  #include <Adafruit_SH110X.h>
  #include <ESP8266WiFi.h>
  #include <ESP8266WebServer.h>
  
  const char* ssid = "Silver";    // Wi-Fi 名稱
  const char* password = "840071003"; // Wi-Fi 密碼
  
  ESP8266WebServer server(80);  // 創建 HTTP 伺服器
  
  #define SCREEN_WIDTH 128 
  #define SCREEN_HEIGHT 64 
  #define OLED_RESET -1
  Adafruit_SH1106G display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
  
  typedef enum {
    START,
    RUNNING,
    GAMEOVER
  } State;
  
  typedef enum {
    LEFT,
    UP,
    RIGHT,
    DOWN
  } Direction;
  
  #define SNAKE_PIECE_SIZE  3
  #define MAX_SANKE_LENGTH 165
  #define MAP_SIZE_X 20
  #define MAP_SIZE_Y 20
  #define STARTING_SNAKE_SIZE 5
  #define SNAKE_MOVE_DELAY 30
  
  State gameState;
  int8_t snake[MAX_SANKE_LENGTH][2];
  uint8_t snake_length;
  Direction dir;
  Direction newDir;
  int8_t fruit[2];
  
  void setup() {
    Serial.begin(9600); // 包綠115200或9600 要在測一下
    Serial.println("\nI2C Scanner");
    display.begin(0x3C);
    Wire.setClock(400000); // 設置 I2C 頻率為 400kHz
  
    // 連接 Wi-Fi
    WiFi.begin(ssid, password);   
    while (WiFi.status() != WL_CONNECTED) {
      delay(1000);
      Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi");
    //開設司服器+三個子路由
    //設定網頁路由
  server.on("/", HTTP_GET, []() {
      String html = "<!DOCTYPE html>";
      html += "<html><head>";
      html += "<title>ESP8266 Snake Game Control</title>";
      html += "<style>";
      html += "body { font-family: Arial, sans-serif; text-align: center; display: flex; flex-direction: column; align-items: center; background: #0d1117; color: #c9d1d9; }";
      html += "h1 { color: #58a6ff; text-shadow: 0 0 10px #58a6ff; }";
      html += ".container { display: flex; justify-content: space-between; align-items: center; width: 80%; margin-top: 20px; border: 1px solid #30363d; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 20px #30363d; background: #161b22; }";
      html += ".left-panel, .right-panel { flex: 1; text-align: center; }";
      html += ".middle-panel { flex: 2; text-align: center; display: flex; justify-content: center; align-items: center; }";
      html += ".controls-grid { display: grid; grid-template-areas: '. up .' 'left start right' '. down restart'; grid-template-columns: 1fr 1fr 1fr; gap: 10px; justify-items: center; }";
      html += "#left { grid-area: left; } #up { grid-area: up; } #right { grid-area: right; } #down { grid-area: down; } #start { grid-area: start; } #restart { grid-area: restart; justify-self: end; }";
      html += "button { margin: 10px; padding: 10px 20px; font-size: 16px; cursor: pointer; width: 100px; height: 50px; background: linear-gradient(45deg, #1f6feb, #3b82f6); color: white; border: none; border-radius: 5px; box-shadow: 0px 4px 10px rgba(59, 130, 246, 0.5); transition: transform 0.2s, box-shadow 0.2s; }";
      html += "button.active { background: #0059d6; transform: scale(1.1); }";
      html += "button:hover { transform: scale(1.05); box-shadow: 0px 6px 15px rgba(59, 130, 246, 0.7); }";
      html += "#timer, #scoreboard { font-size: 24px; color: #58a6ff; border: 1px solid #30363d; padding: 10px; border-radius: 5px; background: #161b22; box-shadow: 0px 0px 10px #58a6ff; }";
      html += "#history { font-size: 20px; margin-top: 20px; color: #00FF00; border: 1px solid #00FF00; padding: 10px; border-radius: 10px; background: #000; box-shadow: 0px 0px 15px #00FF00; display: inline-block; text-align: left; max-width: 300px; }";
      html += "</style></head><body>";
      html += "<h1>Control ESP8266 Snake Game</h1>";
      html += "<div class='container'>";
      html += "<div class='left-panel'><div id='timer'>Timer: 0s</div></div>";
      html += "<div class='middle-panel'><div class='controls-grid'>";
      html += "<button id='up' onclick='sendCommand(2)'>Move Up</button>";
      html += "<button id='left' onclick='sendCommand(0)'>Move Left</button>";
      html += "<button id='start' onclick='startGame()'>Start Game</button>";
      html += "<button id='right' onclick='sendCommand(1)'>Move Right</button>";
      html += "<button id='down' onclick='sendCommand(3)'>Move Down</button>";
      html += "<button id='restart' onclick='restartGame()'>Restart Game</button>";
      html += "</div></div>";
      html += "<div class='right-panel'><div id='scoreboard'>Score: 0</div></div></div>";
      html += "<div id='history'>Command History:<br><span id='history-content'></span></div>";
      html += "<script>";
      html += "let activeButton = null; let timerInterval = null; let timerCount = parseInt(localStorage.getItem('timerCount')) || 0; let score = parseInt(localStorage.getItem('score')) || 0;";
      html += "function updateTimer() { document.getElementById('timer').innerText = Timer: ${timerCount}s; }";
      html += "function updateScore() { document.getElementById('scoreboard').innerText = Score: ${score}; }";
      html += "function resetTimer() { clearInterval(timerInterval); timerCount = 0; localStorage.setItem('timerCount', timerCount); updateTimer(); }";
      html += "function startTimer() { clearInterval(timerInterval); timerInterval = setInterval(() => { timerCount++; localStorage.setItem('timerCount', timerCount); updateTimer(); }, 1000); }";
      html += "function updateHistory(command) { const historyContent = document.getElementById('history-content'); let commandText = ''; switch (command) { case 0: commandText = 'Left'; break; case 1: commandText = 'Right'; break; case 2: commandText = 'Up'; break; case 3: commandText = 'Down'; break; } historyContent.innerHTML += commandText + '<br>'; }";
      html += "function sendCommand(command) { if (activeButton) { activeButton.classList.remove('active'); } switch (command) { case 0: activeButton = document.getElementById('left'); break; case 1: activeButton = document.getElementById('right'); break; case 2: activeButton = document.getElementById('up'); break; case 3: activeButton = document.getElementById('down'); break; } if (activeButton) { activeButton.classList.add('active'); } updateHistory(command); fetch('/move?direction=' + command).then(response => response.text()).then(data => console.log(data)); }";
      html += "function startGame() { startTimer(); fetch('/start_game').then(response => response.text()).then(data => console.log(data)); }";
      html += "function restartGame() { resetTimer(); score = 0; localStorage.setItem('score', score); updateScore(); document.getElementById('history-content').innerHTML = ''; fetch('/restart_game').then(response => response.text()).then(data => console.log(data)); }";
      html += "updateTimer(); updateScore(); if (timerCount > 0) { startTimer(); }";
      html += "</script></body></html>";
      server.send(200, "text/html", html);
  });
  
    // 處理蛇的move
    server.on("/move", HTTP_GET, []() {
      String direction = server.arg("direction");  // 取得 direction 參數(網頁的不是全域) 全域設定是dir
      //0左 1右 2上 3下
      if (direction == "0") {
        newDir = LEFT;
        server.send(200, "text/plain", "Moving Left");
      } else if (direction == "1") {
        newDir = RIGHT;
        server.send(200, "text/plain", "Moving Right");
      } else if (direction == "2") {
        newDir = UP;
        server.send(200, "text/plain", "Moving Up");
      } else if (direction == "3") {
        newDir = DOWN;
        server.send(200, "text/plain", "Moving Down");
      } else {
        server.send(400, "text/plain", "Invalid Command");
      }
    });
  
    // 處理開始遊戲的請求
    server.on("/start_game", HTTP_GET, []() {
      if (gameState == START) {
        gameState = RUNNING;  // 開始遊戲
        Serial.println("Game Started");
      }
      server.send(200, "text/plain", "Game Started");
    });
  
    // 處理重新開始遊戲的請求
    server.on("/restart_game", HTTP_GET, []() {
      if (gameState == GAMEOVER) {
        setupGame();  // 重設遊戲
        gameState = START;  // 回到開始狀態
        Serial.println("Game Restarted");
      }
      server.send(200, "text/plain", "Game Restarted");
    });
  
    server.begin();   // 啟動伺服器
  
    if (!display.begin(0x3C)) { // SH1106 的 I2C 預設地址為 0x3C
      Serial.println(F("SH1106 allocation failed"));
      for (;;) ;
    }
  
    randomSeed(analogRead(A0));//虛擬pin
    setupGame();  // 設定遊戲
  }
  
  void setupGame() {
    gameState = START;
    dir = RIGHT;
    newDir = RIGHT;
    resetSnake();
    generateFruit();
    display.clearDisplay();
    drawMap();
    drawScore();
    drawPressToStart();
    display.display();
  }
  
  void resetSnake() { //打掉重練
    snake_length = STARTING_SNAKE_SIZE;
    for(int i = 0; i < snake_length; i++) {
      snake[i][0] = MAP_SIZE_X / 2 - i;
      snake[i][1] = MAP_SIZE_Y / 2;
    }
  }
  
  int moveTime = 0;//移動時間
  
  void loop() {
    byte error, address;
    int nDevices = 0;
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.println("Scanning...");
  
    for (address = 1; address < 127; address++) {
      Wire.beginTransmission(address);
      error = Wire.endTransmission();
  
      if (error == 0) {
        Serial.print("I2C device found at address 0x");
        if (address < 16)
          Serial.print("0");
        Serial.print(address, HEX);
        Serial.println(" !");
        nDevices++;
      } else if (error == 4) {
        Serial.print("Unknown error at address 0x");
        if (address < 16)
          Serial.print("0");
        Serial.println(address, HEX);
      }
    }
  
    if (nDevices == 0)
      Serial.println("No I2C devices found\n");
    else
      Serial.println("done\n");
  
    delay(10); // 等待 5 秒後重新掃描
    server.handleClient();  // 處理客戶端請求
  
    switch(gameState) {
      case START:
        // 在 START 狀態，等待網頁按鈕來啟動遊戲
        break;
      case RUNNING:
        moveTime++;
        if (moveTime >= SNAKE_MOVE_DELAY) {
          dir = newDir;
          display.clearDisplay();
          if (moveSnake()) {
            gameState = GAMEOVER;
            drawGameover();
            delay(1);
          }
          drawMap();
          drawScore();
          display.display();
          checkFruit();
          moveTime = 0;
        }
        break;
      case GAMEOVER:
        // 遊戲結束，等待restart按鈕重新開始
        break;
    }
    delay(10);//等一下試試這個參數   星星重要
  }
  
  bool moveSnake() {  //開始真的 布林 移動 
    int8_t x = snake[0][0];
    int8_t y = snake[0][1];
    switch(dir) {
      case LEFT:
        x -= 1;
        break;
      case UP:
        y -= 1;
        break;
      case RIGHT:
        x += 1;
        break;
      case DOWN:
        y += 1;
        break;
    }
    if(collisionCheck(x, y))
      return true;
  
    for(int i = snake_length - 1; i > 0; i--) {
      snake[i][0] = snake[i - 1][0];
      snake[i][1] = snake[i - 1][1];
    }
  
    snake[0][0] = x;
    snake[0][1] = y;
    return false;
  }
  
  void checkFruit() { //A0來的
    if(fruit[0] == snake[0][0] && fruit[1] == snake[0][1]) {
      if(snake_length + 1 <= MAX_SANKE_LENGTH)
        snake_length++;
      generateFruit();
    }
  }
  
  void generateFruit() {
    bool b = false;
    do {
      b = false;
      fruit[0] = random(0, MAP_SIZE_X);
      fruit[1] = random(0, MAP_SIZE_Y);
      for(int i = 0; i < snake_length; i++) {
        if(fruit[0] == snake[i][0] && fruit[1] == snake[i][1]) {
          b = true;
          continue;
        }
      }
    } while(b);
  }
  
  bool collisionCheck(int8_t x, int8_t y) {
    for(int i = 1; i < snake_length; i++) {
      if(x == snake[i][0] && y == snake[i][1]) return true;
    }
    if(x < 0 || y < 0 || x >= MAP_SIZE_X || y >= MAP_SIZE_Y) return true;
    return false;
  }
  
  void drawMap() {
    int offsetMapX = SCREEN_WIDTH - SNAKE_PIECE_SIZE * MAP_SIZE_X - 2;
    int offsetMapY = 2;
    display.drawRect(fruit[0] * SNAKE_PIECE_SIZE + offsetMapX, fruit[1] * SNAKE_PIECE_SIZE + offsetMapY, SNAKE_PIECE_SIZE, SNAKE_PIECE_SIZE, SH110X_INVERSE);  // 修正為 SH110X_INVERSE
    display.drawRect(offsetMapX - 2, 0, SNAKE_PIECE_SIZE * MAP_SIZE_X + 4, SNAKE_PIECE_SIZE * MAP_SIZE_Y + 4, SH110X_WHITE);  // 修正為 SH110X_WHITE
    for(int i = 0; i < snake_length; i++) {
      display.fillRect(snake[i][0] * SNAKE_PIECE_SIZE + offsetMapX, snake[i][1] * SNAKE_PIECE_SIZE + offsetMapY, SNAKE_PIECE_SIZE, SNAKE_PIECE_SIZE, SH110X_WHITE);  // 修正為 SH110X_WHITE
    }
  }
  
  void drawScore() {
    display.setTextSize(1);
    display.setTextColor(SH110X_WHITE);  // 修正為 SH110X_WHITE
    display.setCursor(2, 2);
    display.print(F("Score:"));
    display.println(snake_length - STARTING_SNAKE_SIZE);
  }
  
  void drawPressToStart() {
    display.setTextSize(1);
    display.setTextColor(SH110X_WHITE);  // 修正為 SH110X_WHITE
    display.setCursor(2, 20);
    display.print(F("Press \n start \n button \n to start \n game!"));
  }
  
  void drawGameover() {
    display.setTextSize(1);
    display.setTextColor(SH110X_WHITE);  // 修正為 SH110X_WHITE
    display.setCursor(2, 50);
    display.println(F("GAMEOVER"));
  }