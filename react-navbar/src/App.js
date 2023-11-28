// src/App.js
import React, { useState, useEffect } from 'react';
import NavBar from './NavBar';
import axios from 'axios';
import './App.css';

function App() {
  const [usdInr, setUsdInr] = useState(null);
  const [nifty, setNifty] = useState(null);
  const [sensex, setSensex] = useState(null);
  const [toDateNifty, setToDateNifty] = useState('');
  const [toDateSensex, setToDateSensex] = useState('');

  const fetchData = async () => {
    try {
      const apiKey = '2ZI2BPINNZP78HWN'; // Replace with your API key
      const forexUrl = `https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=INR&apikey=${apiKey}`;
      const niftyUrl = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=NSE:NIFTY&apikey=${apiKey}`;
      const sensexUrl = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=BSE:SENSEX&apikey=${apiKey}`;

      const [forexResponse, niftyResponse, sensexResponse] = await Promise.all([
        axios.get(forexUrl),
        axios.get(niftyUrl),
        axios.get(sensexUrl),
      ]);

      setUsdInr(forexResponse.data['Realtime Currency Exchange Rate']['5. Exchange Rate']);
      setNifty(niftyResponse.data['Global Quote']['05. price']);
      setSensex(sensexResponse.data['Global Quote']['05. price']);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const handlePredictNifty = async () => {
    // Validate toDateNifty
    if (!toDateNifty) {
      alert('Please enter a valid To Date for Nifty.');
      return;
    }

    try {
      // Make a POST request to the backend with toDateNifty
      const response = await axios.post('http://localhost:3000/api/predictor', {
        toDate: toDateNifty,
      });

      console.log('Predictor Response (Nifty):', response.data);
    } catch (error) {
      console.error('Error predicting Nifty:', error);
    }
  };

  const handlePredictSensex = async () => {
    // Validate toDateSensex
    if (!toDateSensex) {
      alert('Please enter a valid To Date for Sensex.');
      return;
    }

    try {
      // Make a POST request to the backend with toDateSensex
      const response = await axios.post('http://localhost:3000/api/predictor', {
        toDate: toDateSensex,
      });

      console.log('Predictor Response (Sensex):', response.data);
    } catch (error) {
      console.error('Error predicting Sensex:', error);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <div>
      <NavBar />
      <div className="content-container">
        <div className="column">
          <h2>Nifty Predictor</h2>
          <label>Date:</label>
          <input
            type="date"
            value={toDateNifty}
            onChange={(e) => setToDateNifty(e.target.value)}
          />
          <button onClick={handlePredictNifty}>Submit</button>
        </div>
        <div className="column">
          <h2>Sensex Predictor</h2>
          <label>Date:</label>
          <input
            type="date"
            value={toDateSensex}
            onChange={(e) => setToDateSensex(e.target.value)}
          />
          <button onClick={handlePredictSensex}>Submit</button>
        </div>
        <div className="column">
          <h2>Current Values</h2>
          <p>USD to INR: {usdInr}</p>
          <p>Nifty: {nifty}</p>
          <p>Sensex: {sensex}</p>
        </div>
      </div>
    </div>
  );
}

export default App;
