import React, { useState, useEffect } from "react";
import axios from "axios";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Calendar } from "./ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover";
import { CalendarIcon, RefreshCw, Activity, Thermometer, Droplets, Wind, Gauge } from "lucide-react";
import { format } from "date-fns";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const WeatherDashboard = () => {
  const [weatherSummary, setWeatherSummary] = useState(null);
  const [collectionStatus, setCollectionStatus] = useState(null);
  const [latestWeather, setLatestWeather] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);

  const fetchWeatherData = async () => {
    try {
      setLoading(true);
      
      // Fetch weather summary
      const summaryResponse = await axios.get(`${API}/weather/summary`);
      setWeatherSummary(summaryResponse.data);
      
      // Fetch collection status
      const statusResponse = await axios.get(`${API}/data-collection/status`);
      setCollectionStatus(statusResponse.data);
      
      // Fetch latest weather
      const latestResponse = await axios.get(`${API}/weather/latest`);
      setLatestWeather(latestResponse.data);
      
      setLastUpdate(new Date());
      
    } catch (error) {
      console.error("Error fetching weather data:", error);
    } finally {
      setLoading(false);
    }
  };

  const startDataCollection = async () => {
    try {
      await axios.post(`${API}/data-collection/start`);
      await fetchWeatherData();
    } catch (error) {
      console.error("Error starting data collection:", error);
    }
  };

  const stopDataCollection = async () => {
    try {
      await axios.post(`${API}/data-collection/stop`);
      await fetchWeatherData();
    } catch (error) {
      console.error("Error stopping data collection:", error);
    }
  };

  useEffect(() => {
    fetchWeatherData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchWeatherData();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case "hot": return "bg-red-100 text-red-800";
      case "warm": return "bg-orange-100 text-orange-800";
      case "cool": return "bg-blue-100 text-blue-800";
      case "high": return "bg-red-100 text-red-800";
      case "low": return "bg-blue-100 text-blue-800";
      case "strong": return "bg-red-100 text-red-800";
      case "moderate": return "bg-yellow-100 text-yellow-800";
      case "light": return "bg-green-100 text-green-800";
      case "heavy": return "bg-red-100 text-red-800";
      case "no rain": return "bg-green-100 text-green-800";
      default: return "bg-green-100 text-green-800";
    }
  };

  const getParameterIcon = (parameter) => {
    if (parameter.includes("Temperature")) return <Thermometer className="h-5 w-5" />;
    if (parameter.includes("Humidity")) return <Droplets className="h-5 w-5" />;
    if (parameter.includes("Wind")) return <Wind className="h-5 w-5" />;
    if (parameter.includes("Pressure")) return <Gauge className="h-5 w-5" />;
    return <Activity className="h-5 w-5" />;
  };

  if (loading && !weatherSummary) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-sky-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 text-lg">Loading weather data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <img 
                src="https://upload.wikimedia.org/wikipedia/commons/1/12/Logo_BMKG_%282010%29.png" 
                alt="BMKG Logo" 
                className="h-16 w-16 object-contain"
                data-testid="bmkg-logo"
              />
              <div>
                <h1 className="text-2xl font-bold text-gray-900" data-testid="main-title">
                  BADAN METEOROLOGI KLIMATOLOGI DAN GEOFISIKA
                </h1>
                <p className="text-sm text-gray-600 mt-1" data-testid="subtitle">
                  Automatic Weather Station - System Online
                </p>
                <p className="text-xs text-blue-600 font-medium" data-testid="station-info">
                  AWS Maritim Pontianak
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <Button 
                onClick={fetchWeatherData}
                disabled={loading}
                variant="outline"
                size="sm"
                data-testid="refresh-button"
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900" data-testid="last-update">
                  {lastUpdate ? `Updated: ${format(lastUpdate, 'HH:mm:ss')}` : 'No data'}
                </p>
                <div className="flex items-center space-x-2 mt-1">
                  <div className={`h-2 w-2 rounded-full ${collectionStatus?.collection_active ? 'bg-green-500' : 'bg-red-500'}`}></div>
                  <span className="text-xs text-gray-500" data-testid="collection-status">
                    Collection {collectionStatus?.collection_active ? 'Active' : 'Inactive'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs defaultValue="current" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3 lg:w-[400px]">
            <TabsTrigger value="current" data-testid="current-tab">Current Conditions</TabsTrigger>
            <TabsTrigger value="historical" data-testid="historical-tab">Historical Data</TabsTrigger>
            <TabsTrigger value="predictions" data-testid="predictions-tab">Predictions</TabsTrigger>
          </TabsList>

          {/* Current Conditions Tab */}
          <TabsContent value="current" className="space-y-6">
            {/* Data Collection Controls */}
            <Card data-testid="collection-controls">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center justify-between">
                  <span>Data Collection Control</span>
                  <Badge variant={collectionStatus?.collection_active ? "default" : "secondary"}>
                    {collectionStatus?.collection_active ? "Active" : "Inactive"}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <p className="text-sm text-gray-600">
                      Total Records: <span className="font-medium">{collectionStatus?.total_records || 0}</span>
                    </p>
                    <p className="text-sm text-gray-600">
                      Interval: <span className="font-medium">{collectionStatus?.collection_interval || 60} seconds</span>
                    </p>
                    {collectionStatus?.last_collection && (
                      <p className="text-sm text-gray-600">
                        Last Collection: <span className="font-medium">
                          {format(new Date(collectionStatus.last_collection), 'MMM d, yyyy HH:mm:ss')}
                        </span>
                      </p>
                    )}
                  </div>
                  
                  <div className="flex space-x-2">
                    <Button 
                      onClick={startDataCollection}
                      disabled={collectionStatus?.collection_active}
                      size="sm"
                      data-testid="start-collection-button"
                    >
                      Start Collection
                    </Button>
                    <Button 
                      onClick={stopDataCollection}
                      disabled={!collectionStatus?.collection_active}
                      variant="outline"
                      size="sm"
                      data-testid="stop-collection-button"
                    >
                      Stop Collection
                    </Button>
                  </div>
                </div>
                
                {collectionStatus?.last_error && (
                  <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-sm text-red-600">
                      <strong>Error:</strong> {collectionStatus.last_error}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Weather Parameters Grid */}
            {weatherSummary?.summary && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {weatherSummary.summary.map((param, index) => (
                  <Card key={index} className="relative overflow-hidden" data-testid={`weather-card-${param.parameter.toLowerCase().replace(/\s+/g, '-')}`}>
                    <CardHeader className="pb-2">
                      <CardTitle className="flex items-center justify-between text-sm">
                        <div className="flex items-center space-x-2">
                          {getParameterIcon(param.parameter)}
                          <span className="font-medium">{param.parameter}</span>
                        </div>
                        <Badge className={getStatusColor(param.status)} variant="secondary">
                          {param.status}
                        </Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="space-y-3">
                        <div className="text-center">
                          <div className="text-3xl font-bold text-gray-900" data-testid={`current-value-${param.parameter.toLowerCase().replace(/\s+/g, '-')}`}>
                            {param.current_value.toFixed(1)}
                          </div>
                          <div className="text-sm text-gray-500">{param.unit}</div>
                        </div>
                        
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div className="text-center">
                            <div className="font-medium text-blue-600">{param.min_24h.toFixed(1)}</div>
                            <div className="text-gray-500">Min 24h</div>
                          </div>
                          <div className="text-center">
                            <div className="font-medium text-green-600">{param.avg_24h.toFixed(1)}</div>
                            <div className="text-gray-500">Avg 24h</div>
                          </div>
                          <div className="text-center">
                            <div className="font-medium text-red-600">{param.max_24h.toFixed(1)}</div>
                            <div className="text-gray-500">Max 24h</div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Historical Data Tab */}
          <TabsContent value="historical" className="space-y-6">
            <Card data-testid="historical-data-card">
              <CardHeader>
                <CardTitle>Historical Weather Data</CardTitle>
                <p className="text-sm text-gray-600">
                  View and analyze historical weather patterns and trends
                </p>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12">
                  <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Historical Data Visualization</h3>
                  <p className="text-gray-600 mb-6">
                    Advanced charting and filtering capabilities coming soon
                  </p>
                  <div className="text-sm text-gray-500">
                    Will include: Time series charts, Parameter filtering, Date range selection, Statistical analysis
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Predictions Tab */}
          <TabsContent value="predictions" className="space-y-6">
            <Card data-testid="predictions-card">
              <CardHeader>
                <CardTitle>Weather Predictions</CardTitle>
                <p className="text-sm text-gray-600">
                  AI-powered weather forecasting and analysis
                </p>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Prediction Cards */}
                  {[
                    {
                      title: "Flood Risk Prediction",
                      description: "Water level forecasting for flood prevention",
                      status: "Training",
                      icon: <Droplets className="h-6 w-6" />
                    },
                    {
                      title: "Maritime Weather",
                      description: "Wind speed and direction for marine safety", 
                      status: "Training",
                      icon: <Wind className="h-6 w-6" />
                    },
                    {
                      title: "Rainfall Probability",
                      description: "Local precipitation forecasting",
                      status: "Training", 
                      icon: <Activity className="h-6 w-6" />
                    },
                    {
                      title: "Temperature & Humidity",
                      description: "Urban climate comfort predictions",
                      status: "Training",
                      icon: <Thermometer className="h-6 w-6" />
                    }
                  ].map((prediction, index) => (
                    <Card key={index} className="p-4" data-testid={`prediction-card-${prediction.title.toLowerCase().replace(/\s+/g, '-')}`}>
                      <div className="flex items-start space-x-3">
                        <div className="p-2 bg-blue-100 rounded-lg text-blue-600">
                          {prediction.icon}
                        </div>
                        <div className="flex-1">
                          <h3 className="font-medium text-gray-900 mb-1">{prediction.title}</h3>
                          <p className="text-sm text-gray-600 mb-2">{prediction.description}</p>
                          <Badge variant="secondary" className="text-xs">
                            {prediction.status}
                          </Badge>
                        </div>
                      </div>
                    </Card>
                  ))}
                </div>
                
                <div className="mt-6 text-center py-8 bg-gray-50 rounded-lg">
                  <div className="text-sm text-gray-600">
                    <strong>Deep Learning Models Status:</strong> Currently collecting training data<br/>
                    Prediction models will be available after sufficient data collection
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default WeatherDashboard;