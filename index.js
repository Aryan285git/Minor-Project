const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

app.post('/api/predictor/:symbol', (req, res) => {
  const { symbol } = req.params;
  const { toDate } = req.body;

  // Validate input date
  if (!toDate) {
    return res.status(400).json({ error: 'toDate is required in the request body.' });
  }

  // Your prediction logic can go here for both Nifty and Sensex
  // For now, just sending it back as a response
  res.json({ result: `${symbol === 'nifty' ? 'Nifty' : 'Sensex'} prediction result for ${toDate}` });
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
