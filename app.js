var mnist = require("mnist");
var synaptic = require("synaptic");

var set = mnist.set(700, 20);

var trainingSet = set.training;
var testSet = set.test;


var Layer = synaptic.Layer;
var Network = synaptic.Network;
var Trainer = synaptic.Trainer;

var inputLayer = new Layer(300);
var hiddenLayer = new Layer(200);
var outputLayer = new Layer(10);



inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var myNetwork = new Network({
    input: inputLayer,
    hidden: [hiddenLayer],
    output: outputLayer
});



var trainer = new Trainer(myNetwork);
trainer.train(trainingSet, {
    rate: 0.008,
    iterations: 20,
    error: 0.8,
    shuffle: true,
    log: 1,
    cost: Trainer.cost.CROSS_ENTROPY
});



console.log(myNetwork.activate(testSet[0].input));
console.log(testSet[0].output);
