import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import "./App.css";
import WeatherDashboard from "./components/WeatherDashboard";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<WeatherDashboard />} />
          <Route path="/dashboard" element={<WeatherDashboard />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;