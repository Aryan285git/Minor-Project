// index.js
const express = require('express');
const app = express();
const PORT = 3000;

app.use(express.json());

app.post('/api/predictor', (req, res) => {
  const { toDate } = req.body;

  // Validate input date
  if (!toDate) {
    return res.status(400).json({ error: 'toDate is required in the request body.' });
  }

  // Your logic to process input date can go here
  // For now, just sending it back as a response
  res.json({ toDate });
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
