import React, { useState, useEffect } from 'react';
import Chart from './Chart';
import Filters from './Filters';
import axios from '../services/api';

function Dashboard() {
  const [historicalData, setHistoricalData] = useState([]);

  useEffect(() => {
    async function fetchData() {
      const response = await axios.get('/api/historical');
      setHistoricalData(response.data);
    }
    fetchData();
  }, []);

  return (
    <div>
      <Filters />
      <Chart data={historicalData} />
    </div>
  );
}

export default Dashboard;
