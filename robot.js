var Q = {}, weights = [], alpha = 0.1, gamma = 0.9, epsilon = 0.2;
var optimism = 0.1;  // constant k for exploration function k/(n+1)
var livingCost = 0;
var winFactor = 1000;

var score = 0;

var discPrecision = 1;

// http://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
function shuffle(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
    return array;
}

function round(x, prec) {
  return Number(x).toFixed(prec);
}

function discretize(arr) {
  var out = [];
  for (var i = 0; i < arr.length; i++) {
    out.push(round(arr[i], discPrecision));
  }
  return out;
}

var time = 0;
var iter = 1, totalScore = 0;
function resetGame() {
  totalScore += game.score;
  console.log("Scored " + game.score + " on iteration " + iter +
	      ". New average: " + (totalScore / iter));
  iter++;
  
  game.reset();
  score = 0;
  time = 0;
}

function getState() {
  var state = [game.leftHip.getJointAngle(), game.leftKnee.getJointAngle(),
	       game.rightHip.getJointAngle(), game.rightKnee.getJointAngle(),
	       /*   game.torso.radians */
	       
	       /*game.leftKnee.getBodyA().m_angularVelocity,
	       game.leftKnee.getBodyB().m_angularVelocity,
	       game.rightKnee.getBodyA().m_angularVelocity,
	       game.rightKnee.getBodyB().m_angularVelocity*/];
  
//	       game.leftHip.getJointSpeed(), game.leftKnee.getJointSpeed(),
//	       game.rightHip.getJointSpeed(), game.rightKnee.getJointSpeed()];
  return discretize(state);
}

function initWeights() {
  weights = new Array(getState().length).fill(0);
}

function makeKey(arr) {
  var arr2 = [];
  for (var i = 0; i < arr.length; i++) {
    var v = typeof arr[i] == "string" ?
	round(Number(arr[i]) + 100, discPrecision) :
	arr[i];
    arr2.push(v);
  }
  return arr2.toString();
}

function getActions() {
  return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
	 /* [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0],
	  [0, 1, 0, 1], [0, 0, 1, 1],*/
	  [0, 0, 0, 0]];
}

function randomAction() {
  var actions = getActions();
  return actions[Math.floor(Math.random() * actions.length)];
}

// make defaults for values if they don't exist
function tunnelQValue(s, a) {
  var sKey = makeKey(s), aKey = makeKey(a);
  if (!Q.hasOwnProperty(sKey)) {
    Q[sKey] = {};
  }
  if (!Q[sKey].hasOwnProperty(aKey)) {
    Q[sKey][aKey] = 0;
  }
}

function getQValue(s, a) {
  tunnelQValue(s, a);
  var sKey = makeKey(s), aKey = makeKey(a);
  return Q[sKey][aKey];
}

function setQValue(s, a, value) {
  var sKey = makeKey(s), aKey = makeKey(a);
  tunnelQValue(s, a);
  Q[sKey][aKey] = value;
}

var visits = {};
function visitCount(s, a) {
  var sKey = makeKey(s), aKey = makeKey(a);
  if (visits.hasOwnProperty(sKey) && visits[sKey].hasOwnProperty(aKey)) {
    return visits[sKey][aKey];
  } else {
    return 0;
  }
}
function incVisitCount(s, a) {
  var sKey = makeKey(s), aKey = makeKey(a);
  if (!visits.hasOwnProperty(sKey)) {
    visits[sKey] = {};
  }
  if (!visits[sKey].hasOwnProperty(aKey)) {
    visits[sKey][aKey] = 0;
  }
  visits[sKey][aKey] += 1;
}

function VValue(s) {
  m = -Infinity;
  var actions = getActions();
  for (var i = 0; i < actions.length; i++) {
    m = Math.max(m, getQValue(s, actions[i]));
  }
  return m;
}

// balance exploration and exploitation
function selectAction() {
  var state = getState();
  var actions = (getActions());
  var best = null, bestScore = -Infinity;
  for (var i = 0; i < actions.length; i++) {
    var action = actions[i];
    var score = getQValue(state, action) +
	optimism/(visitCount(state, action) + 1);
    if (score > bestScore) {
      bestScore = score;
      best = action;
    }
  }
  incVisitCount(state, best);
  //console.log(bestScore);
  return best;
}

function bestAction() {
  var state = getState();
  var actions = shuffle(getActions());
  var best = null, bestScore = -Infinity;
  for (var i = 0; i < actions.length; i++) {
    var action = actions[i];
    var QValue = getQValue(state, action);
    if (QValue > bestScore) {
      bestScore = QValue;
      best = action;
    }
  }
  return best;
}

function performAction(action) {
  game.QDown = action[0] == 1;
  game.WDown = action[1] == 1;
  game.ODown = action[2] == 1;
  game.PDown = action[3] == 1;
}

function learnFromAction(s, a, r, sPrime) {
  var curQValue = getQValue(s, a);
  var diff = r + gamma*VValue(sPrime) - curQValue;
  var value = curQValue + alpha*diff;
  setQValue(s, a, value);

  // update weights
  for (var i = 0; i < weights.length; i++) {
    weights[i] += alpha*diff*sPrime[i];  // sPrime???
  }
}

var lastScore = 0, lastState = null, lastAction = null;
var lastScoreCtr = 0;
function learn() {
  var curState = getState();
  if (game.gameEnded) {
    //var reward = time < 30 ? -100 : winFactor * game.score / time;
    learnFromAction(lastState, lastAction, -10, curState);
    resetGame();
  } else {
    if (lastState && lastAction) {
      var reward = (game.score - lastScore)/(lastScoreCtr+1) - livingCost;
      if (game.score != lastScore)
	lastScoreCtr = 0;
      learnFromAction(lastState, lastAction, reward, curState);
    }
    var action;
    action = selectAction();

    /*
    if (Math.random() >= epsilon) {
      action = bestAction();
    } else {
      action = randomAction();
    }
    */
    lastState = curState;
    lastAction = action;
    lastScore = game.score;

    lastScoreCtr++;
    
    performAction(action);

    time++;
  }
}


function contactHook(t) {
  /*
  var a = t.getFixtureA().getBody(), b = t.getFixtureB().getBody();

  var other = null;
  if (a.m_userData == "track") {
    other = b;
  } else if (b.m_userData == "track") {
    other = a;
  }
  //console.log(a.m_userData, b.m_userData);
  if (other) {
    if (other.m_userData == "leftCalf" || other.m_userData == "rightCalf") {
      game.gameEnded = true;
    }
  }*/
}

function main() {
  document.addEventListener("click", function() {
    initWeights();
    setInterval(learn, 1000.0/10);
  });
}

main();
