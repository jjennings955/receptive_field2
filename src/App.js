import React, { Component } from 'react';
import { Stage, Layer, Rect, Text, Group, Transform } from 'react-konva';
import Konva from 'konva';
import * as tf from '@tensorflow/tfjs';
import * as _ from 'underscore';

class Grid extends React.Component {
  state = {
  };

  handleClick = () => {
    this.setState({
      color: Konva.Util.getRandomColor()
    });
  };
  render() {
    return (
      <Group {...this.props} skewY={-27*3.14/180.0}>
      {[...Array(64).keys()].map((i) => (<Rect
        y={Math.floor(i/8)*21}
        x={(i%8)*21}
        width={20}
        height={20}
        fill={ this.props.activityMap[Math.floor(i/8)][i%8] ? "black" : "gray" }
        onMouseOver={() => this.props.onHoverCell(i) }
        onClick={()=>{this.setState({color: "red"})}}
      />))}
      </Group>
    );
  }
}


class App extends Component {
  state = {
    hoveredLayer: 1,
    over: 0,
    result: null
  }
  constructor(props) {
    super(props);
    this.NUM_LAYERS = 5;
    this.CANVAS_SIZE = [128, 128];
    this.SHAPE = [16,16,1];
    this.input = tf.input({shape:this.SHAPE})
    this.layers = []
    for (var i = 0; i < this.NUM_LAYERS; i++)
    {
      this.layers.push(tf.layers.conv2dTranspose({kernelSize:[3,3], filters: 1, useBias:false}))
    }
    
    this.outputs = [this.input, this.layers[0].apply(this.input)]
    
    for (var i = 2; i < this.NUM_LAYERS; i++)
    {
      this.outputs.push(this.layers[i].apply(this.outputs[i-1]));
    }
    
    this.model = tf.model({inputs: this.input, outputs: this.outputs});
    this.foo = _.throttle(this.runModel, 50);
    this.runModel();
    //this.setState({result: tf.zeros([5, 16, 16, 1]).arraySync() });
  }
  hoveredLayer(i) {
    this.setState({hoveredLayer:i})
  }
  runModel() {
    console.log("running model");
    let e = this.state.over;
    var pixel = [Math.floor(e/8), e % 8];
    // craft an input with a single pixel set to 1 to propagate through network
    // this will cause the output of each layer to be the receptive field for that pixel in the previous layer
    var myInput = tf.oneHot(tf.tensor1d([pixel[0]*16 + pixel[1]], 'int32'),
                this.SHAPE[0]*this.SHAPE[1]*this.SHAPE[2]).reshape([1, ...this.SHAPE]).cast('float32')
    console.log("state", this.state);
    console.log("over", e);
    console.log("model", this.model);
    var result;
    if (this.model) {
      result = this.model.predict(myInput);
      if (result)
      {
        console.log("res", result);
        this.result = result;
      }
    }
  }
  
  render() {
    // Stage is a div container
    // Layer is actual canvas element (so you may have several canvases in the stage)
    // And then we have canvas shapes inside the Layer
    
    this.foo();
    let layers = [4, 3, 2, 1, 0];
    return (
      <Stage width={1920} height={800}>
        <Layer>
          <Text text={"Try click on rect" + this.state.hoveredLayer + " " + this.state.over}/>
          {layers.map((i, j) => (<Grid x={170*i}
                                    y={150}
                                    cellHovered={1}
                                    onMouseOver={() => this.hoveredLayer(i)}
                                    activityMap={this.result[j].notEqual(0).squeeze().arraySync() }
                                    hovered={this.state.hoveredLayer == i ? this.state.over : -1 } 
                                    onHoverCell={(i) => { this.setState({over: i}) }} />
                                    ))}
          
          </Layer>
      </Stage>
    );
  }
}

export default App